import json

with open(r"TREC_Washington_Post_collection.v2.jl", encoding="utf8") as fp:
	line = fp.readline()
	dic = json.loads(line)
	# print(line)
	# print((dic))
	for (key,values) in dic.items():
		print(key,":",values,"\n")
		# print(len(values),type(values))

		# for i in len(values):
		# try:



'''
   while line:
       # print("Line {}: {}".format(cnt, line.strip()))
       # print(json.loads(line.strip()))
       # dline=dict(line)
       print((line))
       line = fp.readline()
       cnt += 1
       if cnt > 1:
         break
'''

# parse_json = json.loads(r"TREC_Washington_Post_collection.v2.jl", encoding="utf8")
# print(parse_json)