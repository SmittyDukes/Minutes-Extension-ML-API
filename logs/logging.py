import json
def log_decision_event(event):
    with open("decision_log.jsonl","a") as f:
        f.write(json.dumps(event)+"\n")
#this opens log file, appends event, writes JSON record