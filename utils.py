from datetime import datetime, timedelta

def add_timestamps(timestamp1, timestamp2):
    # Parse the timestamps into datetime objects
    dt1 = datetime.strptime(timestamp1, "%H:%M:%S.%f")
    dt2 = datetime.strptime(timestamp2, "%H:%M:%S.%f")

    # Convert the datetime objects to timedelta
    td1 = timedelta(hours=dt1.hour, minutes=dt1.minute, seconds=dt1.second, microseconds=dt1.microsecond)
    td2 = timedelta(hours=dt2.hour, minutes=dt2.minute, seconds=dt2.second, microseconds=dt2.microsecond)

    # Add the timedeltas
    td_sum = td1 + td2

    # Convert the timedelta back into a datetime object
    dt_sum = datetime(1, 1, 1) + td_sum

    # Format the new datetime back into a string
    new_timestamp = dt_sum.strftime("%H:%M:%S.%f")[:-3]  # remove the last 3 digits of microseconds

    return new_timestamp


def convert_wav_offset_to_timestamp(wav_offset, sampling_rate):
    duration_seconds = wav_offset / sampling_rate
    hours = duration_seconds // 3600
    minutes = (duration_seconds % 3600) // 60
    seconds = duration_seconds % 60
    return f"{int(hours):02}:{int(minutes):02}:{seconds:.3f}"