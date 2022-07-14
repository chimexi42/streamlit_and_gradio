import csv
import datetime



path = 'google_stocks.csv'
file = open(path)
# for line in file:
#     print(line)

lines = [line for line in open(path)]
print(lines)
print(lines[0])
print(lines[1])

print(lines[0].strip().split(','))

dataset = [line.strip().split(',') for line in open(path)]

print(dataset[0])
print(dataset[1])

# using csv module
print(dir(csv))
file = open(path, newline="")
reader = csv.reader(file)
header = next(reader)
print(header)
# data = [row for row in reader]
# print(data[0])

data = []
# for row in reader:
#     # date = datetime.strptime(row[0], '%m/%d/%y')
#     open_price = float(row[1])
#     high = float(row[2])
#     low = float(row[3])
#     close = float(row[4])
#     volume = float(row[5])
#     adj_close = float(row[6])
#
#     data.append([date, open_price, high, low, close, volume, adj_close])

print()

print(dir(datetime))

gvr = datetime.datetime(1995, 1, 31)

print(gvr)
print(gvr.month)
print(gvr.day)

mill = datetime.date(2000, 1,1)
dt = datetime.timedelta(100)

print(mill +dt)

print(gvr.strftime("%A, %B, %Y"))
message ="GVR was born on {:%A, %B, %Y}."

print(message.format(gvr))

launch_date = datetime.date(2017, 3,30)
launch_time =  datetime.time(22,27,0)

launch_datetime = datetime.datetime(2017, 3,30, 22, 27, 0)

print(launch_datetime)
print(launch_date)
print(launch_time)


now = datetime.datetime.today()
print(now)
print(now.microsecond)

moon_landing = "7/20/1969"
moon_landing_datetime = datetime.datetime.strptime(moon_landing, "%m/%d/%Y")
print(moon_landing_datetime)