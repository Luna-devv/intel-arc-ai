import secrets
import string
from io import BytesIO


def save_image(bytes:  BytesIO, model: str):
    chars = string.ascii_letters + string.digits
    password = ''.join(secrets.choice(chars) for i in range(32))

    filename = f"{model}/{password}.png"

    with open(f"./output/{filename}", "wb") as binary_file:
        binary_file.write(bytes)

    return filename
