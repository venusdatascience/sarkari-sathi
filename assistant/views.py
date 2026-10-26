from django.shortcuts import render, redirect
import traceback

from .ai_service import answer_question


def home(request):
    if request.GET.get("new_chat") == "1":
        request.session.pop("chat_history", None)
        return redirect("/")

    chat_history = request.session.get("chat_history", [])
    error = None

    if request.method == "POST":
        question = request.POST.get("question", "").strip()
        language = request.POST.get("language", "ne")

        if language not in ("ne", "en"):
            language = "ne"

        if not question:
            error = "Please enter a question."
        else:
            try:
                result = answer_question(
                    question,
                    language=language,
                )

                chat_history.append({
                    "question": question,
                    "answer": result.get("answer", ""),
                    "evidence": result.get("evidence", []),
                })

                request.session["chat_history"] = chat_history
                request.session.modified = True

            except Exception as e:
                print("\n===== SARKARI SATHI ERROR =====")
                traceback.print_exc()
                print("===============================\n")
                error = f"Error: {type(e).__name__}: {e}"

    return render(
        request,
        "assistant/chat.html",
        {
            "chat_history": chat_history,
            "error": error,
        },
    )
