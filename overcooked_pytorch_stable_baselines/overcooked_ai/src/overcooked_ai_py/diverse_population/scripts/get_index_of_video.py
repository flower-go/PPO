def get_index(input_a, input_b):
    index = (input_a)*30 + input_b + 1
    return index

tuple_list=[(3,25),(2,9),(12,9),(4,9),(5,9),(3,9),(2,8),(12,21),(4,10),(5,5),(8,9),(1,9),(8,0),(9,13),(0,6)]
res_list = []
for t in tuple_list:
    r_i = get_index(t[0], t[1])
    res_list.append(r_i)

print(sorted(res_list))

for i,value in enumerate(res_list):
    print(f"{tuple_list[i]} : {value}")
