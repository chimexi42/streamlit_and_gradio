print(0.1)

print(type(0.1))

# def sqsum1():
# 	return sum(x**2 if x > 0 for x in nums)
#
# def sqsum2():
#   	return sum(x^2 for x in nums if x > 0)
#
# def sqsum3():
#   	return sum(for x in nums if x > 0 then x^2)
#
# def sqsum4():
#   	return sum(x**2 for x in nums if x > 0)
#
# def sqsum5():
#   	return sum(x^2 if x > 0 for x in nums)


class FunEvent:
    def __init__(self, tags, year):
        self.tags = tags
        self.year = year

    def __str__(self):
        return f"FunEvent(tags={self.tags}, year={self.year})"


tags = ["google", "ml"]
year = 2022
bootcamp = FunEvent(tags, year)
tags.append("bootcamp")
year = 2023
print(bootcamp)




def f1(list_of_list):
    result = []
    for inner_list in list_of_list:
        for x in inner_list:
            if x not in result:
                result.append(x)
    return result

def f2(list_of_list):
    flat_list = []
    for inner_list in list_of_list:
        flat_list.extend(inner_list)
    return [
        x for i, x in enumerate(flat_list)
        if flat_list.index(x) == i]

def f3(list_of_list):
    result = []
    seen = set()
    for inner_list in list_of_list:
        for x in inner_list:
            if x not in seen:
                result.append(x)
                seen.add(x)
    return result








# Python one liners
a =20
b =10

a,b = b,a

print(a)
print(b)

l = [1,2,3,4,5]
print(l[::-1])
print(l)

l.reverse()
print(l)
m = [1,3,2,5,5,5,2,2,5,4]
mode = max(set(m), key= m.count)
print(mode)

# print(m.count)

print(set(m))
print(m.count(5))

# multiple line assignment
a,b, c = 4.4, "Awesome", 7
print(a)
print(b)
print(c)

print(type(int('27')))

s = '27'
a = int(s)

print(a+1)

l = ['12', '23', '34']
items = [int(x) for x in l]
print(l)
for item in items:
    print((item, type(item)))

print(4**2)
print(4**.5)
print(16**.5)


print(27**(1/3))
print(abs(-27))

pi = 3.1415
print(round(pi, 2))

print(list(range(7,21)))

l = [1,2,3,4,5,8,9,34]
print(sum(l)/ len(l))


x = 29
v = 10 if x < 20 else 20

print(v)

l = [[1,2], [3,4], [5,6]]
flatten = [j for i in l for j in i ]
print(flatten)

for i in l:
    for j in i:
        print(j)

# palindrome
word = "racecar"
print(word == word[::-1])
word = "foobar"
print(word == word[::-1])

# working with date in pandas
import pandas as pd
dates = ['2015-03-07', 'Mar 7, 2015', '03/07/2017', '2017.03.07', '2017/03/07', '20170307']
print(dates)

print(pd.to_datetime(dates))

date_time = ('2015-03-07 2:30:00 PM', 'Mar 7, 2015 14:30:00', '03/07/2017', '2017.03.07', '2017/03/07', '20170307')

print(date_time)
print(pd.to_datetime(date_time))
print(pd.to_datetime('2-9-2017', dayfirst=True))

print(pd.to_datetime('2015-05-15', format='%Y-%m-%d'))
print(pd.to_datetime('1997-03-07', format='%Y/%m/%d'))

print(pd.to_datetime(['2015-03-07', 'Mar 7, 2015','123'], errors='ignore'))
print(pd.to_datetime(['2015-03-07', 'Mar 7, 2015','123'], errors='coerce'))

epoch_time = 1190155004
print(pd.to_datetime(epoch_time, unit='s'))

epoch_time = 1190155004675
print(pd.to_datetime(epoch_time, unit='ms'))


from datetime import datetime

date1 = "December 8, 2011"
date2 = '8/10/29'
date3 = '09-30-1999 12:23:03'

date1_obj = datetime.strptime(date1, "%B %d, %Y")
date1_obj2 = datetime.strptime(date1, "%B %d, %Y")
print()
print(date1_obj)
print(date1_obj2)

print(type(date1_obj))

date2_obj = datetime.strptime(date2, "%d/%m/%y")
print(date2_obj)

date3_obj = datetime.strptime(date3, "%m-%d-%Y %H:%M:%S")
print(date3_obj)

print()
print("converting back to strings")
print(date1_obj.strftime("%d %B, %Y"))
print(date2_obj.strftime('%B'))
print(date3_obj.strftime("%M:%S"))

#
import  datetime
string_date = "2018-01-13 18:40:51"
format = "%Y-%m-%d %H:%M:%S"
print()
datetime_object = datetime.datetime.strptime(string_date, format)
print("Datetime:",datetime_object)
print("Hour:", datetime_object.hour)
print("Minute:", datetime_object.minute)
print("Seconds:", datetime_object.second)

string_back = datetime_object.strftime("%H:%M, %d-%B-%y")
print(string_back)

















































































































































































































































































































































































































































































































































































































































































































































































































































































































