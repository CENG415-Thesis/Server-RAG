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
    elif event_type == "ADMIN":
        logging.critical(message)
    else:
        logging.debug(message)  # Varsayılan olarak debug seviyesinde log tut


