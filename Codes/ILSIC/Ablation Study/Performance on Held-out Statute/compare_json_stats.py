import json

file1 = '/home/shounak/HDD/Layman-LSI/dataset/court_tmp_filtered_statutes.json'  
file2 = '/home/shounak/HDD/Layman-LSI/dataset/test_filtered_statutes.json'       
intersection_outfile = '/home/shounak/HDD/Layman-LSI/dataset/court&test_filtered_statutes.json' 

def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def analyze_json_intersections(file1, file2, intersection_outfile):
    
    data1 = load_json(file1)
    data2 = load_json(file2)
    
    
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())

    
    intersection = keys1 & keys2
    intersection_dict = {k: data1[k] for k in intersection}

  
    save_json(intersection_outfile, intersection_dict)

   
    print(f"Intersection count: {len(intersection)}")
    print(f"Intersection saved to {intersection_outfile}")

if __name__ == '__main__':
    analyze_json_intersections(file1, file2, intersection_outfile)
