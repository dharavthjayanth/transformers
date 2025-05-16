import json
import os

def get_user_access(email: str) -> list:
    access_file = os.path.join(os.path.dirname(__file__), "users_access.json")

    with open(access_file, "r") as f:
        access_data = json.load(f)

    email = email.lower()
    if email not in access_data:
        raise ValueError(f"Access not defined for: {email}")

    return access_data[email]["access"]
