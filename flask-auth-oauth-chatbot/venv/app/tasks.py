# app/tasks.py
from celery import shared_task
from app import create_app
from langgraph_agent import run_agent_turn
# Import necessary models if needed inside task

flask_app = create_app()
flask_app.app_context().push()

@shared_task(bind=True)
def run_agent_task(self, conversation_id: int, user_query: str):
    """Celery task to run the LangGraph agent asynchronously."""
    print(f"Celery task received for Conv {conversation_id}")
    try:
        # run_agent_turn saves messages directly now
        _ = run_agent_turn(conversation_id, user_query)
        # How to notify user? WebSockets/SSE needed here.
        # This task currently doesn't return the result directly to the route.
    except Exception as e:
        print(f"Celery task failed for Conv {conversation_id}: {e}")
        # Log error, maybe add system error message to chat?
        # raise self.retry(exc=e) # Optional retry