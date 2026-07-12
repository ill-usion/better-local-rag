import json
import asyncio
from uuid import uuid4
from collections.abc import Generator
from rag import make_app
from flask import Flask, request, session, g, make_response, render_template, Response

app = Flask(__name__)
app.secret_key = "<CHANGE THIS>"
app.template_folder = 'static'


def get_rag_app():
    """
    Returns the RAG app global state
    """
    if "rag_app" not in g:
        g.rag_app = make_app()

    return g.rag_app


async def call_model(state: dict, session_id: str) -> Generator[dict, None, None]:
    '''
    Runs the RAG agent loop

    Args:
        state: Chat state
        session_id: Current session id used for chat sessions

    Yields:
        dict: Current generation status indicating retrieval status or new chat chunks
    '''
    with app.app_context():
        rag_app = get_rag_app()
        async for event in rag_app.astream_events(
                state,
                config={"configurable": {"thread_id": session_id}},
                return_only_outputs=True
            ):
            if event["event"] == "on_chat_model_start":
                yield { "event": "chat_model_start", "data": f"Chat model started." }

            if event["event"] == "on_chat_model_stream" and len(event["data"]["chunk"].content):
                yield { "event": "chunk", "data": event["data"]["chunk"].content }

            if event["event"] == "on_chat_model_end":
                yield { "event": "chat_model_end", "data": f"Chat model endded." }

            if event["event"] == "on_retriever_start":
                yield {"event": "retrieval_start", "data": "Starting retrieval..." }

            if event["event"] == "on_retriever_end":
                yield { "event": "retrieval_end", "data": f"Retrieval ended with {len(event["data"]["output"])} results." }


def __iter_over_async(async_generator) -> Generator[dict, None, None]:
    """
    Iterates over the async iterable and yields formatted chunks.
    
    Yields:
        dict: Generation and chunk status 
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    iterator = async_generator.__aiter__()

    async def get_next() -> tuple[bool, Any]:
        """
        Retrieves the next chunk from the iterator.

        Returns:
            tuple[bool, Any]: A tuple with a boolean indicating if the iteration is done and the chunk.
        """
        try:
            obj = await iterator.__anext__()
            return False, obj
        except StopAsyncIteration:
            return True, None
        except Exception as e:
            print(f"Error in get_next: {e}")
            # Handle exceptions from callable_fn or chunk_fnc
            return True, None

    try:
        while True:
            done, obj = loop.run_until_complete(get_next())
            if done:
                break
            yield obj
    finally:
        loop.close()


def generate_stream(callable_fn, chunk_fnc) -> Generator[dict, None, None]:
    """
    Generates a stream from `callable_fn` and processes it using `chunk_fnc`

    Returns:
        `Generator[dict, None, None]`
    """

    async def run_inference():
        async for chunk in callable_fn():
            yield chunk

    async def collect_chunks():
        chunks = []
        async for chunk in run_inference():
            yield chunk_fnc(chunk) 

    return __iter_over_async(collect_chunks())


def send_chunk(chunk):
    """
    Process chunk
    """
    # Convert dict to json string
    return json.dumps(chunk)


@app.route('/')
def index():
    session_id = request.cookies.get('session_id')
    if not session_id:
        session_id = str(uuid4())
        response = make_response(render_template('index.html'))
        response.set_cookie('session_id', session_id)
        return response

    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('input')
    if not user_input:
        return {"error": "No input provided"}, 400

    session_id = request.cookies.get('session_id')
    if not session_id:
        return {"error": "No session ID found in cookies"}, 400

    state = {
        "input": user_input
    }

    return Response(
        generate_stream(lambda: call_model(state, session_id), send_chunk),
        mimetype="application/json"
    )

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)