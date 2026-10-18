import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path

import bcrypt
from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, abort
)
from supabase import create_client

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

from rag_engine import retrieve_chunks, generate_answer, prepare_sources


app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.getenv("SESSION_SECRET") or os.getenv("FLASK_SECRET_KEY"),
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("COOKIE_SECURE", "false").lower() == "true",
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Activity logs may require elevated server-side permissions depending on RLS.
# The service key is only used on the server and must never be sent to the browser.
supabase_memory = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    supabase_memory = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
elif supabase is not None:
    # This fallback works only if your Supabase RLS policies permit the operations.
    supabase_memory = supabase


@app.before_request
def check_configuration_and_csrf():
    if not app.config["SECRET_KEY"]:
        if request.endpoint != "health":
            abort(500, description="SESSION_SECRET is not configured.")
    if request.method == "POST":
        token = session.get("_csrf_token")
        supplied = request.form.get("_csrf_token", "")
        if not token or not secrets.compare_digest(token, supplied):
            abort(400, description="Invalid form token. Refresh the page and try again.")


@app.context_processor
def inject_csrf_token():
    def csrf_token():
        if "_csrf_token" not in session:
            session["_csrf_token"] = secrets.token_urlsafe(32)
        return session["_csrf_token"]

    # The sidebar is shared by the dashboard and course pages. Keep the query
    # user-scoped and fail gracefully if the chat table is temporarily unavailable.
    sidebar_chat_sessions = []
    current_user = session.get("user")
    if current_user and supabase_memory is not None:
        try:
            sidebar_chat_sessions = get_user_chat_sessions(current_user["id"], limit=30)
        except Exception:
            app.logger.exception("Could not load sidebar chat history")

    return {"csrf_token": csrf_token, "sidebar_chat_sessions": sidebar_chat_sessions}


def require_supabase():
    if supabase is None:
        raise RuntimeError(
            "Supabase is not configured. Set SUPABASE_URL and SUPABASE_KEY in your .env."
        )


def current_memory_enabled():
    # Memory is enabled by default for each signed-in browser session.
    return bool(session.get("memory_enabled", True))


def get_user_learning_memory(user_id, limit=10):
    """Return only this user's recent learning activities."""
    if not current_memory_enabled() or supabase_memory is None:
        return []
    try:
        result = (
            supabase_memory.table("activity_logs")
            .select("activity_type, activity_data, created_at")
            .eq("user_id", user_id)
            .eq("activity_type", "ASK_QUESTION")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception:
        app.logger.exception("Could not retrieve learning memory")
        return []


def log_learning_activity(user_id, activity_type, activity_data):
    """Save activity only when the user has enabled memory."""
    if not current_memory_enabled() or supabase_memory is None:
        return False
    try:
        supabase_memory.table("activity_logs").insert({
            "user_id": user_id,
            "activity_type": activity_type,
            "activity_data": activity_data,
        }).execute()
        return True
    except Exception:
        # Don't break a successful AI answer if activity logging fails.
        app.logger.exception("Could not save learning activity")
        return False


def load_course_videos():
    """Read course titles and YouTube URLs from the existing urls.txt file."""
    urls_path = BASE_DIR / "urls.txt"
    if not urls_path.exists():
        raise FileNotFoundError("urls.txt was not found in the Flask project folder.")

    lines = [line.strip() for line in urls_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    videos = []
    for index, line in enumerate(lines):
        match = re.match(r"^(\d+)\)\s*(.+)$", line)
        if not match:
            continue

        number = int(match.group(1))
        full_title = match.group(2).strip()
        youtube_url = None
        for following_line in lines[index + 1:]:
            if re.match(r"^\d+\)", following_line):
                break
            if following_line.startswith(("https://", "http://")):
                youtube_url = following_line
                break
        if not youtube_url:
            continue

        # Keep the human-readable lesson title, without the repeated playlist suffix.
        title = full_title.split("|", 1)[0].strip()
        videos.append({"number": number, "name": title or full_title, "youtube_url": youtube_url})

    videos.sort(key=lambda video: video["number"])
    return videos


def get_user_video_progress(user_id):
    """Return completed video numbers from the user's latest progress events.

    Progress is stored in activity_logs with its own activity_type, independently
    of the Student Learning Memory toggle.
    """
    if supabase_memory is None:
        raise RuntimeError("Supabase activity logging is not configured.")

    result = (
        supabase_memory.table("activity_logs")
        .select("activity_data, created_at")
        .eq("user_id", user_id)
        .eq("activity_type", "VIDEO_PROGRESS")
        .order("created_at", desc=True)
        .limit(1000)
        .execute()
    )

    latest_state = {}
    for entry in result.data or []:
        data = entry.get("activity_data") or {}
        if not isinstance(data, dict):
            continue
        try:
            video_number = int(data.get("video_number"))
        except (TypeError, ValueError):
            continue
        if video_number not in latest_state:
            latest_state[video_number] = bool(data.get("completed", False))

    return {number for number, completed in latest_state.items() if completed}


def save_video_progress(user_id, video_number, completed):
    """Save a progress event without depending on the memory-enabled setting."""
    if supabase_memory is None:
        raise RuntimeError("Supabase activity logging is not configured.")

    supabase_memory.table("activity_logs").insert({
        "user_id": user_id,
        "activity_type": "VIDEO_PROGRESS",
        "activity_data": {
            "video_number": int(video_number),
            "completed": bool(completed),
        },
    }).execute()


def format_memory_for_prompt(entries):
    """Convert Supabase activity rows into readable prompt context."""
    if not entries:
        return "No previous learning activity is available."

    lines = []
    for entry in entries:
        activity_type = str(entry.get("activity_type") or "ACTIVITY")
        activity_data = entry.get("activity_data") or {}

        if isinstance(activity_data, dict):
            question = activity_data.get("question")
            if question:
                line = f"- {activity_type}: Student previously asked: {question}"
                video_numbers = activity_data.get("video_numbers")
                if video_numbers:
                    line += f" (related videos: {video_numbers})"
                lines.append(line)
            else:
                lines.append(f"- {activity_type}: {activity_data}")
        else:
            lines.append(f"- {activity_type}: {activity_data}")

    return "\n".join(lines) if lines else "No previous learning activity is available."


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user"):
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def find_user_by_email(email):
    require_supabase()
    result = (
        supabase.table("users")
        .select("id, name, email, password_hash")
        .eq("email", email)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None



def get_chat_database():
    """Return the server-side Supabase client used for chat storage."""
    if supabase_memory is None:
        raise RuntimeError("Supabase chat storage is not configured.")
    return supabase_memory


def get_user_chat_sessions(user_id, limit=50):
    """Return only the signed-in user's recent conversations."""
    result = (
        get_chat_database().table("chat_sessions")
        .select("id, title, created_at, updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


def get_owned_chat_session(chat_id, user_id):
    """Verify that a conversation belongs to the current user."""
    result = (
        get_chat_database().table("chat_sessions")
        .select("id, title, created_at, updated_at")
        .eq("id", chat_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


def create_chat_session(user_id, first_question):
    """Create a conversation whose title is based on its first question."""
    title = " ".join((first_question or "New Chat").split())[:60] or "New Chat"
    result = (
        get_chat_database().table("chat_sessions")
        .insert({"user_id": user_id, "title": title})
        .execute()
    )
    if not result.data:
        raise RuntimeError("Supabase did not return the new chat session ID.")
    return result.data[0]


def save_chat_message(chat_id, user_id, role, content):
    """Persist one user or assistant message."""
    if role not in {"user", "assistant"}:
        raise ValueError("Invalid chat message role.")
    get_chat_database().table("messages").insert({
        "session_id": chat_id,
        "user_id": user_id,
        "role": role,
        "content": content,
    }).execute()


def get_chat_messages(chat_id, user_id, limit=500):
    """Fetch messages only after the parent chat has been ownership-checked."""
    result = (
        get_chat_database().table("messages")
        .select("id, role, content, created_at")
        .eq("session_id", chat_id)
        .eq("user_id", user_id)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return result.data or []


def touch_chat_session(chat_id, user_id):
    """Move the active chat to the top of the history list."""
    get_chat_database().table("chat_sessions").update({
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", chat_id).eq("user_id", user_id).execute()


def render_dashboard(question="", answer=None, sources=None, recommended_video=None,
                     rag_error=None, show_memory=False, memory_entries=None,
                     chat_sessions=None, active_chat=None, chat_messages=None):
    user = session["user"]
    if chat_sessions is None:
        try:
            chat_sessions = get_user_chat_sessions(user["id"])
        except Exception:
            app.logger.exception("Could not load chat history")
            chat_sessions = []
    return render_template(
        "dashboard.html",
        user=user,
        question=question,
        answer=answer,
        sources=sources or [],
        recommended_video=recommended_video,
        rag_error=rag_error,
        memory_enabled=current_memory_enabled(),
        show_memory=show_memory,
        memory_entries=memory_entries or [],
        chat_sessions=chat_sessions or [],
        active_chat=active_chat,
        chat_messages=chat_messages or [],
    )


@app.get("/health")
def health():
    return {"status": "ok", "app": "Python RAG Teaching Assistant - Flask"}


@app.get("/")
def index():
    if session.get("user"):
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if session.get("user"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or not email or not password:
            flash("Please fill in all required fields.", "error")
        elif len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
        elif password != confirm_password:
            flash("Passwords do not match.", "error")
        else:
            try:
                require_supabase()
                if find_user_by_email(email):
                    flash("This email is already registered. Please log in.", "error")
                else:
                    password_hash = bcrypt.hashpw(
                        password.encode("utf-8"), bcrypt.gensalt()
                    ).decode("utf-8")
                    result = supabase.table("users").insert({
                        "name": name,
                        "email": email,
                        "password_hash": password_hash,
                    }).execute()

                    if result.data:
                        flash("Account created. You can now log in.", "success")
                        return redirect(url_for("login"))
                    flash("Account could not be created. Check your database setup.", "error")
            except Exception as exc:
                app.logger.exception("Signup failed")
                flash(
                    f"Signup failed. Check Supabase configuration and the users table. ({type(exc).__name__})",
                    "error"
                )

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Enter your email and password.", "error")
        else:
            try:
                user = find_user_by_email(email)
                if user and bcrypt.checkpw(
                    password.encode("utf-8"),
                    user["password_hash"].encode("utf-8"),
                ):
                    session.clear()
                    session.permanent = True
                    session["user"] = {
                        "id": user["id"],
                        "name": user["name"],
                        "email": user["email"],
                    }
                    session["memory_enabled"] = True
                    session["_csrf_token"] = secrets.token_urlsafe(32)
                    flash("Welcome back!", "success")
                    return redirect(url_for("dashboard"))
                flash("Invalid email or password.", "error")
            except Exception as exc:
                app.logger.exception("Login failed")
                flash(
                    f"Login failed. Check Supabase configuration and the users table. ({type(exc).__name__})",
                    "error"
                )

    return render_template("login.html")


@app.get("/dashboard")
@login_required
def dashboard():
    user_id = session["user"]["id"]

    # A deliberate New Chat request clears the active conversation.
    if request.args.get("new_chat") == "1":
        session.pop("active_chat_id", None)
        return render_dashboard()

    # Explicitly selected sidebar chat wins; otherwise resume the current chat.
    requested_chat_id = request.args.get("chat_id", "").strip()
    if requested_chat_id:
        session["active_chat_id"] = requested_chat_id
    else:
        requested_chat_id = str(session.get("active_chat_id", "") or "").strip()

    active_chat = None
    chat_messages = []

    if requested_chat_id:
        try:
            active_chat = get_owned_chat_session(requested_chat_id, user_id)
            if active_chat is None:
                session.pop("active_chat_id", None)
                if request.args.get("chat_id"):
                    flash("That conversation was not found in your account.", "warning")
                return redirect(url_for("dashboard"))
            session["active_chat_id"] = active_chat["id"]
            chat_messages = get_chat_messages(active_chat["id"], user_id)
        except Exception as exc:
            app.logger.exception("Could not open chat history")
            flash(f"Chat history could not be loaded ({type(exc).__name__}). Check the Supabase chat tables.", "error")

    return render_dashboard(
        active_chat=active_chat,
        chat_messages=chat_messages,
    )


@app.post("/chat/new")
@login_required
def new_chat():
    # Keep the next question in a new conversation, not the previous one.
    session.pop("active_chat_id", None)
    return redirect(url_for("dashboard", new_chat="1"))


@app.post("/chat/<chat_id>/rename")
@login_required
def rename_chat(chat_id):
    """Rename only a conversation owned by the signed-in user."""
    user_id = session["user"]["id"]
    title = " ".join(request.form.get("title", "").split()).strip()
    if not title:
        flash("Chat name cannot be empty.", "warning")
        return redirect(url_for("dashboard", chat_id=chat_id))
    title = title[:80]
    try:
        chat = get_owned_chat_session(chat_id, user_id)
        if chat is None:
            abort(404)
        get_chat_database().table("chat_sessions").update({
            "title": title,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", chat_id).eq("user_id", user_id).execute()
        flash("Chat renamed successfully.", "success")
    except Exception:
        app.logger.exception("Could not rename chat")
        flash("Could not rename this chat. Please try again.", "error")
    return redirect(url_for("dashboard", chat_id=chat_id))


@app.post("/chat/<chat_id>/delete")
@login_required
def delete_chat(chat_id):
    """Delete a user's conversation and its messages after ownership check."""
    user_id = session["user"]["id"]
    try:
        chat = get_owned_chat_session(chat_id, user_id)
        if chat is None:
            abort(404)

        # Delete child messages first to avoid foreign-key constraint errors.
        get_chat_database().table("messages").delete().eq(
            "session_id", chat_id
        ).eq("user_id", user_id).execute()
        get_chat_database().table("chat_sessions").delete().eq(
            "id", chat_id
        ).eq("user_id", user_id).execute()

        if str(session.get("active_chat_id", "")) == str(chat_id):
            session.pop("active_chat_id", None)
        flash("Chat deleted.", "success")
    except Exception:
        app.logger.exception("Could not delete chat")
        flash("Could not delete this chat. Please try again.", "error")
    return redirect(url_for("dashboard", new_chat="1"))


@app.post("/memory/toggle")
@login_required
def toggle_memory():
    new_state = not current_memory_enabled()
    session["memory_enabled"] = new_state
    if new_state:
        flash("🧠 Student Learning Memory is enabled.", "success")
    else:
        flash("Student Learning Memory is off. New activity will not be saved or used for personalization.", "warning")
    return redirect(url_for("dashboard"))




@app.get("/student-dashboard")
@login_required
def student_dashboard():
    """Show user-scoped learning stats without changing existing memory settings."""
    user_id = session["user"]["id"]
    stats = {"questions": 0, "activities": 0, "completed_videos": 0, "total_videos": 100,
             "daily_labels": [], "daily_questions": [], "activity_labels": [], "activity_counts": []}
    recent_activities = []
    warning = None
    try:
        if supabase_memory is None:
            raise RuntimeError("Supabase activity logging is not configured.")
        rows = (supabase_memory.table("activity_logs")
                .select("activity_type, activity_data, created_at")
                .eq("user_id", user_id).order("created_at", desc=True).limit(1000).execute().data or [])
        # Count the latest VIDEO_PROGRESS state for each video, not every toggle event.
        latest_video_state = {}
        from collections import Counter
        from datetime import date
        question_days = Counter()
        activity_types = Counter()
        for row in rows:
            kind = str(row.get("activity_type") or "ACTIVITY")
            data = row.get("activity_data") or {}
            if not isinstance(data, dict):
                data = {}
            activity_types[kind] += 1
            if kind == "ASK_QUESTION":
                stats["questions"] += 1
                created = str(row.get("created_at") or "")[:10]
                if created:
                    question_days[created] += 1
            elif kind == "VIDEO_PROGRESS":
                try:
                    number = int(data.get("video_number"))
                    if number not in latest_video_state:
                        latest_video_state[number] = bool(data.get("completed", False))
                except (TypeError, ValueError):
                    pass
            if kind == "ASK_QUESTION" and len(recent_activities) < 6:
                recent_activities.append({"type": kind, "question": str(data.get("question") or "Question")[:180], "created_at": row.get("created_at", "")})
        stats["activities"] = len(rows)
        stats["completed_videos"] = sum(1 for value in latest_video_state.values() if value)
        stats["progress_percent"] = round(stats["completed_videos"] * 100 / max(stats["total_videos"], 1))
        from datetime import datetime, timedelta
        today = datetime.now().date()
        days = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
        stats["daily_labels"] = [day.strftime("%b %d") for day in days]
        stats["daily_questions"] = [question_days.get(day.isoformat(), 0) for day in days]
        stats["activity_labels"] = list(activity_types.keys())[:8]
        stats["activity_counts"] = [activity_types[key] for key in stats["activity_labels"]]
    except Exception as exc:
        app.logger.exception("Could not build student dashboard statistics")
        warning = f"Some statistics could not be loaded ({type(exc).__name__}). Check Supabase activity_logs permissions."
        stats["progress_percent"] = 0
    return render_template("student_dashboard.html", user=session["user"], stats=stats,
                           recent_activities=recent_activities, warning=warning)


@app.get("/course")
@login_required
def course_videos():
    try:
        all_videos = load_course_videos()
    except Exception as exc:
        app.logger.exception("Could not load course videos")
        flash(f"Course videos could not be loaded ({type(exc).__name__}). Check that urls.txt is in the project folder.", "error")
        all_videos = []

    try:
        completed_videos = get_user_video_progress(session["user"]["id"])
    except Exception as exc:
        app.logger.exception("Could not load course progress")
        flash(f"Progress could not be loaded ({type(exc).__name__}). Check Supabase permissions and activity_logs.", "error")
        completed_videos = set()

    for video in all_videos:
        video["completed"] = video["number"] in completed_videos

    search_query = request.args.get("q", "").strip()
    if search_query:
        query = search_query.casefold()
        visible_videos = [
            video for video in all_videos
            if query in video["name"].casefold()
            or query in str(video["number"])
        ]
    else:
        visible_videos = all_videos

    total_videos = len(all_videos)
    completed_count = sum(1 for video in all_videos if video["completed"])
    progress_percent = round((completed_count / total_videos) * 100) if total_videos else 0

    return render_template(
        "course.html",
        user=session["user"],
        videos=visible_videos,
        search_query=search_query,
        total_videos=total_videos,
        completed_count=completed_count,
        remaining_count=max(0, total_videos - completed_count),
        progress_percent=progress_percent,
    )


@app.post("/course/progress")
@login_required
def toggle_video_progress():
    try:
        video_number = int(request.form.get("video_number", ""))
    except (TypeError, ValueError):
        abort(400, description="Invalid video number.")

    try:
        videos = load_course_videos()
    except Exception as exc:
        app.logger.exception("Could not load course videos for progress update")
        flash(f"Course videos could not be loaded ({type(exc).__name__}). Check urls.txt.", "error")
        return redirect(url_for("course_videos", q=request.form.get("q", "").strip() or None))

    valid_numbers = {video["number"] for video in videos}
    if video_number not in valid_numbers:
        abort(404, description="Course video not found.")

    try:
        completed_videos = get_user_video_progress(session["user"]["id"])
        new_state = video_number not in completed_videos
        save_video_progress(session["user"]["id"], video_number, new_state)

        if new_state:
            flash(f"Video {video_number} marked as completed!", "success")
        else:
            flash(f"Video {video_number} marked as not completed.", "success")
    except Exception as exc:
        app.logger.exception("Could not update course progress")
        flash(f"Progress could not be saved ({type(exc).__name__}). Check Supabase permissions and activity_logs.", "error")

    return redirect(url_for("course_videos", q=request.form.get("q", "").strip() or None))


@app.get("/memory")
@login_required
def view_memory():
    if not current_memory_enabled():
        flash("Turn on Student Learning Memory to view your saved learning activity.", "warning")
        return redirect(url_for("dashboard"))

    entries = get_user_learning_memory(session["user"]["id"], limit=20)
    return render_dashboard(show_memory=True, memory_entries=entries)


@app.post("/ask")
@login_required
def ask():
    user = session["user"]
    user_id = user["id"]
    question = request.form.get("question", "").strip()
    requested_chat_id = request.form.get("chat_id", "").strip()
    if not requested_chat_id:
        requested_chat_id = str(session.get("active_chat_id", "") or "").strip()

    if not question:
        flash("Please enter a Python question first.", "warning")
        return redirect(url_for("dashboard", chat_id=requested_chat_id or None))
    if len(question) > 2000:
        flash("Please keep your question under 2,000 characters.", "error")
        return redirect(url_for("dashboard", chat_id=requested_chat_id or None))

    active_chat = None
    user_message_saved = False
    assistant_message_saved = False
    try:
        if requested_chat_id:
            active_chat = get_owned_chat_session(requested_chat_id, user_id)
            if active_chat is None:
                abort(404, description="Chat not found.")
        else:
            active_chat = create_chat_session(user_id, question)

        chat_id = active_chat["id"]
        session["active_chat_id"] = chat_id
        save_chat_message(chat_id, user_id, "user", question)
        user_message_saved = True
        touch_chat_session(chat_id, user_id)

        results = retrieve_chunks(question, top_k=3)
        memory_entries = get_user_learning_memory(user_id, limit=10) if current_memory_enabled() else []
        memory_text = format_memory_for_prompt(memory_entries)
        answer = generate_answer(question, results, memory_text=memory_text)
        sources = prepare_sources(results)
        recommended_video = sources[0] if sources else None

        # Keep chat history independent of the Student Learning Memory toggle.
        save_chat_message(chat_id, user_id, "assistant", answer)
        assistant_message_saved = True
        touch_chat_session(chat_id, user_id)

        # Memory controls personalization/activity logging only; it does not control chat history.
        if current_memory_enabled():
            video_numbers = [source["number"] for source in sources]
            log_learning_activity(
                user_id,
                "ASK_QUESTION",
                {"question": question, "video_numbers": video_numbers},
            )

        chat_messages = get_chat_messages(chat_id, user_id)
        chat_sessions = get_user_chat_sessions(user_id)
        return render_dashboard(
            question=question,
            answer=answer,
            sources=sources,
            recommended_video=recommended_video,
            chat_sessions=chat_sessions,
            active_chat=active_chat,
            chat_messages=chat_messages,
        )
    except Exception as exc:
        app.logger.exception("RAG question or chat history operation failed")
        if isinstance(exc, (FileNotFoundError, RuntimeError, ValueError)):
            message = str(exc)
        else:
            message = f"The request failed ({type(exc).__name__}). Check the terminal for details."

        chat_messages = []
        chat_sessions = None
        if active_chat is not None:
            try:
                if user_message_saved and not assistant_message_saved:
                    # Leave the question visible in history even if the AI service failed.
                    save_chat_message(
                        active_chat["id"], user_id, "assistant",
                        "Sorry, I couldn't generate an answer this time. Please try again."
                    )
                chat_messages = get_chat_messages(active_chat["id"], user_id)
                chat_sessions = get_user_chat_sessions(user_id)
            except Exception:
                app.logger.exception("Could not load the conversation after an error")
        return render_dashboard(
            question=question,
            rag_error=message,
            chat_sessions=chat_sessions,
            active_chat=active_chat,
            chat_messages=chat_messages,
        ), 200


@app.post("/logout")
@login_required
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


@app.errorhandler(500)
def internal_error(error):
    message = getattr(error, "description", "Something went wrong.")
    return render_template("error.html", code=500, message=message), 500


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
