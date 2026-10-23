"""Generate the intent training set from sentence templates.

    python ml/make_dataset.py        # writes data/intent_train.jsonl

Templates are expanded with random subjects / days / times / teachers, then
augmented with polite prefixes, casing changes and typos.  The *test* set
(eval/intent_heldout.jsonl) is written by hand with different wording so the
score is not measured on sentences the model has effectively seen.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
random.seed(7)

SUBJ = ["DBMS", "Java", "Operating Systems", "OS", "Computer Networks", "Machine Learning", "Software Engineering",
        "Data Structures", "Discrete Mathematics", "Digital Logic", "DBMS lab", "Java lab", "database", "networks"]
DAY = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Mon", "Tue", "Wed", "Thu", "Fri"]
TIME = ["9 AM", "10 AM", "11 am", "noon", "1 PM", "2 PM", "3 pm", "4 PM", "5 PM", "4:30 PM", "3:15 pm"]
TEACH = ["Dr. Rao", "Prof. Mehta", "Mr. Kumar", "Ms. Sharma", "Dr. Iyer", "Rao", "Mehta", "Kumar", "Prof. Gupta"]
PART = ["morning", "afternoon", "evening", "after lunch", "before lunch"]
GROUP = ["CSE 5th semester", "CSE 3rd semester", "section A", "section B", "CSE-5-A", "5th sem", "semester 3"]
NUM = ["four", "five", "six", "seven", "5", "6", "8"]

T = {
 "create": [
  "create a timetable for {group}", "generate a timetable for {group} with {num} subjects",
  "build a new timetable for {group}", "prepare the schedule for {group}", "make a timetable for {group}",
  "I need a timetable for {group}", "please generate a schedule for {group}", "set up the {group} timetable",
  "produce a weekly timetable for {group} with {num} subjects", "can you create a timetable for {group}",
  "draft a timetable for {group}", "come up with a timetable for {group}", "generate timetable {group}",
  "new timetable for {group}", "I want a timetable for {group} with {num} subjects",
 ],
 "modify": [
  "schedule {subj} on {day} {part}", "schedule {subj} on {day}", "keep {day} free", "keep {day} {part} free",
  "don't schedule classes after {time}", "do not schedule anything before {time}", "no classes after {time}",
  "avoid scheduling two lab sessions on the same day", "avoid the first period for {teach}",
  "{teach} is not available on {day}", "{teach} cannot teach on {day}", "put {subj} on {day} {part}",
  "I prefer {subj} in the {part}", "try to schedule {subj} before {time}", "give us {day} off",
  "give us a longer lunch break", "make the lunch break shorter", "no classes between {time} and {time}",
  "{subj} must be on {day}", "ensure {day} {part} is free", "make sure there are no labs on {day}",
  "limit classes to before {time}", "classes only after {time}", "avoid {day} for {teach}",
  "prefer no classes on {day} {part}", "{subj} should be in the {part}", "don't put {subj} on {day}",
  "make {day} a free day", "add a break on {day} {part}", "restrict {teach} to {part} classes",
 ],
 "move": [
  "move {subj} to {day}", "move {subj} from {day} to {day}", "shift {subj} to {day} {part}",
  "reschedule {subj} to {day}", "move the {subj} class from {day} to {day}", "relocate {subj} to {day}",
  "swap {subj} to {day}", "transfer {subj} from {day} to {day}", "postpone {subj} to {day}",
  "shift the {subj} lecture from {day} to {day}", "can you move {subj} to {day}", "please move {subj} to {day} {part}",
  "change {subj} from {day} to {day}", "reschedule {teach}'s class to {day}", "move it to {day}",
  "move {subj} to {day} at {time}", "prepone {subj} to {day}",
 ],
 "view": [
  "show the timetable for {group}", "show only laboratory sessions", "display {teach}'s timetable",
  "show me {day}", "show everything", "view the timetable for {group}", "list the lab classes",
  "show only theory classes", "display the schedule for {group}", "what is the timetable for {teach}",
  "show {subj} classes", "see the {day} schedule", "show the full timetable", "print the timetable for {group}",
  "show me only {day} {part}", "display all labs", "when is {subj}", "which classes are on {day}",
  "show timetable of {teach}", "show the {group} timetable",
 ],
 "regenerate": [
  "regenerate the timetable", "please regenerate", "redo the timetable", "reshuffle everything",
  "try again", "generate it again", "give me another timetable", "I want a different timetable",
  "rerun the solver", "shuffle the classes and try again", "make a new version of this timetable",
  "regenerate it with the same rules", "re-run the scheduling", "another version please",
  "can you reshuffle the timetable", "redo it",
 ],
 "reset": [
  "reset everything", "start over", "clear all constraints", "forget everything", "start afresh",
  "remove all constraints", "reset the timetable", "clear everything and start over", "wipe it and begin again",
  "reset all", "delete all constraints", "let's start from scratch",
 ],
}
PREFIX = ["", "", "", "please ", "can you ", "could you ", "I want you to ", "kindly ", "hey, ", "ok so ", "now "]
SUFFIX = ["", "", "", ".", "!", " please", " for me", " now"]


def typo(s):
    words = s.split()
    i = random.randrange(len(words))
    w = words[i]
    if len(w) > 4:
        j = random.randrange(1, len(w) - 1)
        w = w[:j] + w[j + 1:] if random.random() < 0.5 else w[:j] + w[j] + w[j:]
    words[i] = w
    return " ".join(words)


def fill(t):
    return t.format(subj=random.choice(SUBJ), day=random.choice(DAY), time=random.choice(TIME),
                    teach=random.choice(TEACH), part=random.choice(PART), group=random.choice(GROUP),
                    num=random.choice(NUM)).replace("{day}", random.choice(DAY))


def main():
    rows = []
    for intent, temps in T.items():
        for t in temps:
            for _ in range(6):
                s = random.choice(PREFIX) + fill(t) + random.choice(SUFFIX)
                if random.random() < 0.18:
                    s = typo(s)
                if random.random() < 0.15:
                    s = s.upper() if random.random() < 0.3 else s.capitalize()
                rows.append({"text": s, "intent": intent})
    random.shuffle(rows)
    out = ROOT / "data" / "intent_train.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(len(rows), "sentences ->", out)


if __name__ == "__main__":
    main()
