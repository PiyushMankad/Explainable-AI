import json

with open(r"E:\Intelligent Systems\Dissertation ####\TREC_Washington_Post_collection.v2.jl", encoding="utf8") as fp:
	line = fp.readline()
	count = 1
	while line:
		dic = json.loads(line)
		title = dic['title']
		
		## file writing
		article = open("article{}.txt".format(count),"a+")
		# article.write(title)
		content = ""

		for i,(key,values) in enumerate(dic.items()):
			## outputs all the key and value pairs
			# print(key,":",values,"\n")

			try:
				l = len(dic[key])
				# print(type(dic[key]),len(dic[key]))

				for j in range(len(dic[key])):
					# print(dic[key][0])
					
					for (key2,val2) in dic[key][j].items():
						if key2 == "content":
							## outputs all the content keys
							content = dic[key][j][key2]
							print(key2,":",val2,"\n")
							# print("CONTENT",content)
							# content = content+val2+"\n"
							# article.write(json.dumps(content),"\n")

			except:
				pass

		line = fp.readline()
		count+=1

		article.close()
		print("\n\n\n<<<<<<<<########## NEXT ARTICLE {} ############>>>>>>".format(count))
		if count >=1:
			break
