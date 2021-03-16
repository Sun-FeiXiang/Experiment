def f(a):
    a = a + 1

a = 9
print(a)

def test(a_int, b_list):
    a_int = a_int + 1
    b_list.append('13')
    print('inner a_int:' + str(a_int))
    print('inner b_list:' + str(b_list))

if __name__ == '__main__':
    a_int = 5
    b_list = [10, 11]
    test(a_int, b_list)
    print('outer a_int:' + str(a_int))
    print('outer b_list:' + str(b_list))
