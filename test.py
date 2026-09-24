x = 5
name = 'anya'
nums = [3, 1, 2, 5]
nums1 = [9, 7, 33, 0]

person1 = {'name': 'Anya', 'age': 19}
person2 = {'name': 'Egor', 'age': 20}

print(type(name))
print(nums[0])
print(nums[0])

#print('hello')

def hello(x, y):

    for i in range(0, y):
        print(x)

    while y>1:
        print(y)
        y-=1

    return 10

print('000000')
for i in nums:
    print(i)

print('-------')
for i, j in zip(nums, nums1):
    print(i+j)

k= "hello,        privet, halo"
k = k.split(',')

for i in range(len(k)):
    k[i] = k[i].strip()

print(k)
print(type(k))

with open('titanic.csv', encoding='utf-8') as f:
    header = f.readline().strip().split