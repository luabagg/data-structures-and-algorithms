def secs_to_hours(secs: int) -> tuple:
    return divmod(secs, 3600)

def secs_to_minutes(secs: int) -> tuple:
    return divmod(secs, 60)

def format_seconds(secs: int) -> str:
    hours, secs = secs_to_hours(secs)
    minutes, secs = secs_to_minutes(secs)

    return f"Time: {hours}:{minutes}:{secs}"

seconds = input("Write the number of seconds: ")
if not seconds.isnumeric():
    print("Invalid number")
    exit()

print(format_seconds(int(seconds)))
