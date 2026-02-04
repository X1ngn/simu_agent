from langgraph.checkpoint.memory import MemorySaver
from backend.agent.graph import build_graph

def build_test_app(designer, human_review, worker, analyst):
    return build_graph(
        designer=designer,
        human_review=human_review,
        worker=worker,
        analyst=analyst,
        checkpointer=MemorySaver(),
    )
