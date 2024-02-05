import regex as re
import dspy
from dspy.predict import Retry
from dspy.datasets import HotPotQA

from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dsp.utils import EM, normalize_text
from examples.longformqa.utils import (
    extract_text_by_citation,
    correct_citation_format,
    has_citations,
    citations_check,
)

colbertv2_wiki17_abstracts = dspy.ColBERTv2(
    url="http://20.102.90.50:2017/wiki17_abstracts"
)
# change to qdrant later !!
dspy.settings.configure(rm=colbertv2_wiki17_abstracts)
# turbo = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=500)
turbo = dspy.HFClientVLLM(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1", port=8000, url="http://localhost"
)
dspy.settings.configure(lm=turbo, trace=[], temperature=0.7)

dataset = HotPotQA(
    train_seed=1,
    train_size=100,
    eval_seed=2023,
    dev_size=50,
    test_size=0,
    keep_details=True,
)
trainset = [x.with_inputs("question") for x in dataset.train]
devset = [x.with_inputs("question") for x in dataset.dev]

train_example = trainset[0]
print(f"Question: {train_example.question}")
print(f"Answer: {train_example.answer}")
print(f"Relevant Wikipedia Titles: {train_example.gold_titles}")

dev_example = devset[18]
print(f"Question: {dev_example.question}")
print(f"Answer: {dev_example.answer}")
print(f"Relevant Wikipedia Titles: {dev_example.gold_titles}")

from dsp.utils import deduplicate


class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()


class GenerateCitedParagraph(dspy.Signature):
    """Generate a paragraph with citations."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    paragraph = dspy.OutputField(desc="includes citations")


class LongFormQA(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()
        self.generate_query = [
            dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)
        ]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_cited_paragraph = dspy.ChainOfThought(GenerateCitedParagraph)
        self.max_hops = max_hops

    def forward(self, question):
        context = []
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
        pred = self.generate_cited_paragraph(context=context, question=question)
        pred = dspy.Prediction(context=context, paragraph=pred.paragraph)
        return pred


class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""

    context = dspy.InputField(desc="may contain relevant facts")
    text = dspy.InputField(desc="between 1 to 2 sentences")
    faithfulness = dspy.OutputField(
        desc="boolean indicating if text is faithful to context"
    )


def citation_faithfulness(example, pred, trace):
    paragraph, context = pred.paragraph, pred.context
    citation_dict = extract_text_by_citation(paragraph)
    if not citation_dict:
        return False, None
    context_dict = {str(i): context[i].split(" | ")[1] for i in range(len(context))}
    faithfulness_results = []
    unfaithful_citations = []
    check_citation_faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
    for citation_num, texts in citation_dict.items():
        if citation_num not in context_dict:
            continue
        current_context = context_dict[citation_num]
        for text in texts:
            try:
                result = check_citation_faithfulness(context=current_context, text=text)
                is_faithful = result.faithfulness.lower() == "true"
                faithfulness_results.append(is_faithful)
                if not is_faithful:
                    unfaithful_citations.append(
                        {
                            "paragraph": paragraph,
                            "text": text,
                            "context": current_context,
                        }
                    )
            except ValueError as e:
                faithfulness_results.append(False)
                unfaithful_citations.append(
                    {"paragraph": paragraph, "text": text, "error": str(e)}
                )
    final_faithfulness = all(faithfulness_results)
    if not faithfulness_results:
        return False, None
    return final_faithfulness, unfaithful_citations


def extract_cited_titles_from_paragraph(paragraph, context):
    cited_indices = [int(m.group(1)) for m in re.finditer(r"\[(\d+)\]\.", paragraph)]
    cited_indices = [index - 1 for index in cited_indices if index <= len(context)]
    cited_titles = [context[index].split(" | ")[0] for index in cited_indices]
    return cited_titles


def calculate_recall(example, pred, trace=None):
    gold_titles = set(example["gold_titles"])
    found_cited_titles = set(
        extract_cited_titles_from_paragraph(pred.paragraph, pred.context)
    )
    intersection = gold_titles.intersection(found_cited_titles)
    recall = len(intersection) / len(gold_titles) if gold_titles else 0
    return recall


def calculate_precision(example, pred, trace=None):
    gold_titles = set(example["gold_titles"])
    found_cited_titles = set(
        extract_cited_titles_from_paragraph(pred.paragraph, pred.context)
    )
    intersection = gold_titles.intersection(found_cited_titles)
    precision = len(intersection) / len(found_cited_titles) if found_cited_titles else 0
    return precision


def answer_correctness(example, pred, trace=None):
    assert hasattr(example, "answer"), "Example does not have 'answer'."
    normalized_context = normalize_text(pred.paragraph)
    if isinstance(example.answer, str):
        gold_answers = [example.answer]
    elif isinstance(example.answer, list):
        gold_answers = example.answer
    else:
        raise ValueError("'example.answer' is not string or list.")
    return (
        1
        if any(normalize_text(answer) in normalized_context for answer in gold_answers)
        else 0
    )


def evaluate(module):
    correctness_values = []
    recall_values = []
    precision_values = []
    citation_faithfulness_values = []
    for i in range(len(devset)):
        example = devset[i]
        print("Evaluating example", i, "of", len(devset))
        try:
            pred = module(question=example.question)
            correctness_values.append(answer_correctness(example, pred))
            citation_faithfulness_score, _ = citation_faithfulness(None, pred, None)
            citation_faithfulness_values.append(citation_faithfulness_score)
            recall = calculate_recall(example, pred)
            precision = calculate_precision(example, pred)
            recall_values.append(recall)
            precision_values.append(precision)
        except Exception as e:
            print(f"Failed generation with error: {e}")

    average_correctness = (
        sum(correctness_values) / len(devset) if correctness_values else 0
    )
    average_recall = sum(recall_values) / len(devset) if recall_values else 0
    average_precision = sum(precision_values) / len(devset) if precision_values else 0
    average_citation_faithfulness = (
        sum(citation_faithfulness_values) / len(devset)
        if citation_faithfulness_values
        else 0
    )

    print(f"Average Correctness: {average_correctness}")
    print(f"Average Recall: {average_recall}")
    print(f"Average Precision: {average_precision}")
    print(f"Average Citation Faithfulness: {average_citation_faithfulness}")


longformqa = LongFormQA()
evaluate(longformqa)

question = devset[6].question
pred = longformqa(question)
citation_faithfulness_score, _ = citation_faithfulness(None, pred, None)

print(f"Question: {question}")
print(f"Predicted Paragraph: {pred.paragraph}")
print(f"Citation Faithfulness: {citation_faithfulness_score}")


class LongFormQAWithAssertions(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()
        self.generate_query = [
            dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)
        ]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_cited_paragraph = dspy.ChainOfThought(GenerateCitedParagraph)
        self.max_hops = max_hops

    def forward(self, question):
        context = []
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
        pred = self.generate_cited_paragraph(context=context, question=question)
        pred = dspy.Prediction(context=context, paragraph=pred.paragraph)
        dspy.Suggest(
            citations_check(pred.paragraph),
            f"Make sure every 1-2 sentences has citations. If any 1-2 sentences lack citations, add them in 'text... [x].' format.",
            target_module=GenerateCitedParagraph,
        )
        _, unfaithful_outputs = citation_faithfulness(None, pred, None)
        if unfaithful_outputs:
            unfaithful_pairs = [
                (output["text"], output["context"]) for output in unfaithful_outputs
            ]
            for _, context in unfaithful_pairs:
                dspy.Suggest(
                    len(unfaithful_pairs) == 0,
                    f"Make sure your output is based on the following context: '{context}'.",
                    target_module=GenerateCitedParagraph,
                )
        else:
            return pred
        return pred


from dspy.primitives.assertions import assert_transform_module, backtrack_handler

longformqa_with_assertions = assert_transform_module(
    LongFormQAWithAssertions().map_named_predictors(Retry), backtrack_handler
)
evaluate(longformqa_with_assertions)

question = devset[6].question
pred = longformqa_with_assertions(question)
citation_faithfulness_score, _ = citation_faithfulness(None, pred, None)

print(f"Question: {question}")
print(f"Predicted Paragraph: {pred.paragraph}")
print(f"Citation Faithfulness: {citation_faithfulness_score}")

# longformqa = LongFormQA()
# teleprompter = BootstrapFewShotWithRandomSearch(
#     metric=answer_correctness, max_bootstrapped_demos=2, num_candidate_programs=3
# )
# cited_longformqa = teleprompter.compile(
#     student=longformqa, teacher=longformqa, trainset=trainset, valset=devset[:100]
# )
# evaluate(cited_longformqa)
# 
# longformqa = LongFormQA()
# teleprompter = BootstrapFewShotWithRandomSearch(
#     metric=answer_correctness, max_bootstrapped_demos=2, num_candidate_programs=3
# )
# cited_longformqa_teacher = teleprompter.compile(
#     student=longformqa,
#     teacher=assert_transform_module(
#         LongFormQAWithAssertions().map_named_predictors(Retry), backtrack_handler
#     ),
#     trainset=trainset,
#     valset=devset[:100],
# )
# evaluate(cited_longformqa_teacher)

longformqa = LongFormQA()
teleprompter = BootstrapFewShotWithRandomSearch(
    metric=answer_correctness, max_rounds=3, max_bootstrapped_demos=6, num_candidate_programs=3
)
cited_longformqa_student_teacher = teleprompter.compile(
    student=assert_transform_module(
        LongFormQAWithAssertions().map_named_predictors(Retry), backtrack_handler
    ),
    teacher=assert_transform_module(
        LongFormQAWithAssertions().map_named_predictors(Retry), backtrack_handler
    ),
    trainset=trainset,
    valset=devset[:100],
)
evaluate(cited_longformqa_student_teacher)
