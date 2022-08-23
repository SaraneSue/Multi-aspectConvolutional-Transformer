import csv
def parse(s):
    parts = s.split('\n')
    for part in parts:
        splits = part.split('：')
        if len(splits)>1:
            print(splits[1])

def table(path):
    with open(path, newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar=' ')
        for row in spamreader:
            print('&'.join(row)+'\\\\')

def parsematrix(s):
    with open('table.csv', 'w') as f:
        rightsum = 0
        all = 0
        i = 0
        for row in s.split('\n'):
            row = row.replace('[', '')
            row = row.replace(']', '')
            row = row.replace('   ', ' ')
            row = row.replace('  ', ' ')
            row = [d for d in row.split(' ') if len(d)>0]
            row_nums = [int(d) for d in row]
            row_sum = 0
            for rowitem in row_nums:
                row_sum += rowitem
            print('&'+'&'.join(row+[str(round(row_nums[i]/row_sum*100, 2))])+'\\\\')
            rightsum += row_nums[i]
            all += row_sum
            i+=1
        print('& '*10+str(round(rightsum/all*100, 2)))





if __name__ == '__main__':
    s= '''  [[258   1   3   0   3   0   0   8   1   0]
 [  0 269   1   0   1   0   0   0   0   3]
 [  0   3 187   0   1   0   2   0   0   2]
 [  0   0   0 272   0   0   0   0   2   0]
 [  0   0   0   0 196   0   0   0   0   0]
 [  0   0   2   0   5 188   0   0   0   0]
 [  0   0   1   0   0   0 195   0   0   0]
 [  1   0   0   0   0   0   0 267   3   2]
 [  0   0   0   2   0   0   0   3 269   0]
 [  0   0   0   2   0   0   0   0   0 272]]'''
    parsematrix(s)