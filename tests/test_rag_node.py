"""RAG workflow-node tests using a fake retriever."""

import agent.nodes as nodes


class FakeRetriever:
    device = "cpu"

    def __init__(self):
        self.queries = []

    def search(
        self,
        query,
        *,
        top_k,
        deduplicate_pages,
    ):
        self.queries.append(query)

        return [
            {
                "rank": 1,
                "score": 0.9,
                "text": "Mock IQA evidence.",
                "metadata": {
                    "source_path": "mock/paper.pdf",
                    "page_number": 1,
                },
                "citation": "mock/paper.pdf#page=1",
            }
        ]


def test_single_image_rag_query(monkeypatch):
    retriever = FakeRetriever()

    monkeypatch.setattr(
        nodes,
        "_CONFIG",
        {
            "rag": {
                "top_k": 4,
                "runtime_device": "cpu",
            }
        },
    )

    monkeypatch.setattr(
        nodes,
        "get_retriever",
        lambda *, device: retriever,
    )

    result = nodes.retrieve_knowledge(
        {
            "route": "single_image",
            "user_query": "分析图像质量",
            "cot_iqa_result": {
                "diagnosis": {
                    "gaussian_blur": "severe",
                },
                "restoration_suggestion": (
                    "apply deblurring"
                ),
            },
        }
    )

    assert len(result["rag_context"]) == 1
    assert result["execution_trace"] == [
        "retrieve_knowledge"
    ]
    assert "gaussian_blur" in retriever.queries[0]
    assert "single image quality assessment" in (
        retriever.queries[0]
    )


def test_comparison_rag_query(monkeypatch):
    retriever = FakeRetriever()

    monkeypatch.setattr(
        nodes,
        "_CONFIG",
        {
            "rag": {
                "top_k": 4,
                "runtime_device": "cpu",
            }
        },
    )

    monkeypatch.setattr(
        nodes,
        "get_retriever",
        lambda *, device: retriever,
    )

    result = nodes.retrieve_knowledge(
        {
            "route": "comparison",
            "user_query": "比较两张图片",
            "comparison_result": {
                "winner_item_id": "image_1",
                "confidence": "medium",
            },
            "comparison_cot_results": {
                "image_1": {
                    "diagnosis": "JPEG compression",
                },
                "image_2": {
                    "diagnosis": "Gaussian blur",
                },
            },
        }
    )

    assert len(result["rag_context"]) == 1
    assert "paired image quality comparison" in (
        retriever.queries[0]
    )
    assert "winner_item_id" in retriever.queries[0]
