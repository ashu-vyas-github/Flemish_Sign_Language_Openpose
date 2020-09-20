import csv

def create_submission(test_probabilities, path):

    with open(path, 'w') as out:
        writer = csv.writer(out, delimiter=',')
        writer.writerow(['Id','Predicted'])
        for idx in range(test_probabilities.shape[0]):
            writer.writerow([idx, test_probabilities[idx]])