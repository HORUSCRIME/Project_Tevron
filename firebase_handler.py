# firebase_handler.py

import firebase_admin
from firebase_admin import credentials, db
import logging

class FirebaseHandler:

    def __init__(self, credentials_path='firebase_credentials.json', db_url='https://tevron-a668c-default-rtdb.asia-southeast1.firebasedatabase.app/'):

        self.logger = logging.getLogger(__name__)
        self.ref = None
        try:
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred, {'databaseURL': db_url})
            self.ref = db.reference('sensor_data')
            self.logger.info("Successfully connected to Firebase.")
        except FileNotFoundError:
            self.logger.error(f"Firebase credentials file not found at '{credentials_path}'.")
            self.logger.error("Please ensure 'firebase_credentials.json' is in the root directory.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Firebase: {e}")

    def send_data(self, data):

        if not self.ref:
            self.logger.warning("Firebase is not initialized. Cannot send data.")
            return False

        try:
            self.ref.push(data)
            self.logger.info("Successfully sent data record to Firebase.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send data to Firebase: {e}")
            return False

    def get_all_data(self):

        if not self.ref:
            self.logger.warning("Firebase is not initialized. Cannot retrieve data.")
            return None
        
        try:
            return self.ref.get()
        except Exception as e:
            self.logger.error(f"Failed to retrieve data from Firebase: {e}")
            return None


