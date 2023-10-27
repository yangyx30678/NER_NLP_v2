with open('./test-submit.txt','r') as f:
    lines = f.readlines()

with open('./example.txt','w') as f:
    for line in lines:
        if line=='\n': f.write(line)
        else:
            f.write(line.replace('\n','\tO\n'))
