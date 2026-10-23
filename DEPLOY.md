# Deployment notes

## Option A - Docker (recommended for a college server)
```bash
docker build -t timetable-nlp .
docker run -p 8501:8501 -v timetable_data:/app/data timetable-nlp
```
Open http://localhost:8501. The `-v` flag keeps your academic data (`data/academic_data.json`) and saved versions (`data/versions/`) when the container restarts. Without it they are lost.

## Option B - Streamlit Community Cloud (free, public demo)
1. Put the project folder in a GitHub repository (the `requirements.txt` must be at the top level).
2. On share.streamlit.io choose the repo and `app.py` as the main file, Python 3.11.
3. Know the limits: the disk is **temporary** (edited data and saved versions disappear when the app restarts or sleeps), the app is public unless you restrict viewers, and the free tier has little memory. The solver time limit slider (default 10 s) is fine; avoid raising it above 30 s there.

## Option C - Run on a lab PC
`pip install -r requirements.txt` then `streamlit run app.py --server.address 0.0.0.0` makes it reachable from other PCs on the same network at `http://<this-pc-ip>:8501` (allow the port in the firewall).

## Things to know before real use
- **There is no login.** The "View as" selector only changes which pages are shown; it is not security. Anyone who can open the URL can switch to Admin. Put the app behind your college VPN / reverse-proxy authentication if the data matters.
- Every visitor shares the same files in `data/`. Two admins editing at once will overwrite each other.
- First start builds the intent model in about a second; no model download is needed.
