import os
import africastalking
import ssl
import urllib3
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings()


_initialized = False

def _init():
    global _initialized
    if not _initialized:
        africastalking.initialize(
            username=os.environ["AT_USERNAME"],
            api_key=os.environ["AT_API_KEY"],
        )
        _initialized = True


def send_otp_sms(phone: str, code: str) -> dict:
    """
    Send a 6-digit OTP via SMS using Africa's Talking.

    Args:
        phone: full phone number with country code, e.g. "+237612345678"
        code:  6-digit OTP string

    Returns:
        dict with keys: success (bool), message (str), raw (dict)
    """
    _init()

    sender_id = os.environ.get("AT_SENDER_ID") or None
    message = f"Your SafeMum AI verification code is: {code}. It expires in 5 minutes. Do not share it with anyone."

    try:
        sms = africastalking.SMS
        response = sms.send(
            message=message,
            recipients=[phone],
            sender_id=sender_id,
        )

        # Africa's Talking returns a nested response — check the status
        recipients = response.get("SMSMessageData", {}).get("Recipients", [])
        if recipients:
            status = recipients[0].get("status", "")
            status_code = recipients[0].get("statusCode", 0)
            cost = recipients[0].get("cost", "")

            if status == "Success" or status_code == 101:
                print(f"[SafeMum SMS] OTP sent to {phone} | cost: {cost}")
                return {"success": True, "message": "OTP sent", "raw": response}
            else:
                print(f"[SafeMum SMS] Delivery failed for {phone}: {status}")
                return {"success": False, "message": f"Delivery failed: {status}", "raw": response}

        return {"success": False, "message": "No recipients in response", "raw": response}

    except Exception as e:
        print(f"[SafeMum SMS] Error sending to {phone}: {str(e)}")
        return {"success": False, "message": str(e), "raw": {}}