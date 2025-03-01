import logging
# Logları kaydetmek için dosya ayarı
logging.basicConfig(
    filename="logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)




def log_event(event_type: str, message: str):
    """
    event_type: INFO, ERROR, WARNING gibi log türleri
    message: Loglanacak mesaj
    """
    if event_type == "INFO":
        logging.info(message)
    elif event_type == "ERROR":
        logging.error(message)
    elif event_type == "WARNING":
        logging.warning(message)
    else:
        logging.debug(message)  # Varsayılan olarak debug seviyesinde log tut



import json

def log_state_change(state):
    with open("state_changes.json", "a", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
        f.write("\n")
    log_event("INFO", f"State changed: {state}")
