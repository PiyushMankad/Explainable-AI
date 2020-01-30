import json

with open(r"TREC_Washington_Post_collection.v2.jl", encoding="utf8") as fp:
	line = fp.readline()
	dic = json.loads(line)
	# print(line)
	# print((dic))
	for i,(key,values) in enumerate(dic.items()):
		print(key,":",values,"\n")
		try:
			l = len(dic[key])
			# print(type(dic[key]),len(dic[key]))

			for j in range(len(dic[key])):
				# print(dic[key][0])
				for (key2,val2) in dic[key][j].items():
					print(key2,":",val2,"\n")

				print("###############")


		except:
			pass
