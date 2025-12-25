import re, os, time
from rich.progress import track

def extract_pathology_sections(report_path):
    ### No report: 11-020966 12-007545 15-013753 16-009512 16-020595
    microscopic_findings = []
    checklist = {}
    
    extracting_microscopic = False
    extracting_checklist = False
    
    current_sentence = ""
    current_checklist_item = ""
    current_checklist_number = None
    f = open(report_path, 'r+', encoding='UTF-8')
    for line in f.readlines():    
        line = line.strip()
        
        if "MICROSCOPIC FINDING" in line:
            extracting_microscopic = True
            continue
        
        if "Colorectal Cancer Checklist" in line:
            extracting_microscopic = False
            extracting_checklist = True
            if current_sentence:
                microscopic_findings.append(current_sentence.strip())
                current_sentence = ""
            continue
        
        if "Pathologists" in line or line == "":
            extracting_microscopic = False
            extracting_checklist = False
            if current_sentence:
                microscopic_findings.append(current_sentence.strip())
                current_sentence = ""
            if current_checklist_number is not None:
                checklist[current_checklist_number] = current_checklist_item.strip()
                current_checklist_item = ""
                current_checklist_number = None
            continue
        
        if extracting_microscopic:
            current_sentence += (" " if current_sentence else "") + line
        
        if extracting_checklist:
            match = re.match(r"(\d+)\.\s(.+)", line)
            if match:
                if current_checklist_number is not None:
                    checklist[current_checklist_number] = current_checklist_item.strip()
                current_checklist_number = int(match.group(1))
                current_checklist_item = match.group(2).strip()
            else:
                current_checklist_item += (" " if current_checklist_item else "") + line

    if microscopic_findings != []:
        tmp_str = microscopic_findings[0].strip().split('.')
        microscopic_findings = [x for x in tmp_str if x != '']

    return microscopic_findings, checklist

def parse_report():
    cases_path = 'H:/Pathology/'
    error_cases = []
    final_result = {}
    for c in track(os.listdir(cases_path)):
        p = os.path.join(cases_path, c)
        if 'clinical_record' in c:
            continue
        findings, checklist = '', ''
        # print("Current case: ", p)
        for file in os.listdir(p):
            if '.txt' in file:
                tmp_p = os.path.join(p, file)
                findings, checklist = extract_pathology_sections(tmp_p)

        # print("Cases: ", c)
        # print("Findings: ", findings)
        # print("Checklist: ", checklist)
        # time.sleep(10)
        if len(findings) != 0 and len(checklist) != 0:
            final_result[str(c)] = {"Findings" : findings, "Checklist" : checklist} 
        else:
            error_cases.append(p)
    
    return final_result

if __name__ == '__main__':

    cases_path = 'H:/Pathology/'
    ind = 0
    error_cases = []
    final_result = {}
    for c in track(os.listdir(cases_path)):
        p = os.path.join(cases_path, c)
        if 'clinical_record' in c:
            continue
        findings, checklist = '', ''
        # print("Current case: ", p)
        for file in os.listdir(p):
            if '.txt' in file:
                tmp_p = os.path.join(p, file)
                findings, checklist = extract_pathology_sections(tmp_p)

        # print("Cases: ", c)
        # print("Findings: ", findings)
        # print("Checklist: ", checklist)
        # time.sleep(10)
        if findings != [] and checklist != {}:
            final_result[str(c)] = {"Findings" : findings, "Checklist" : checklist} 
            ind += 1
        else:
            error_cases.append(p)
    # print(len(final_result))
    tumor_diff = []
    tumor_diff_count = {'well' : 0, 'moderately' : 0, 'poorly' : 0}
    for k, v in final_result.items():
        # print("Report number: ", k)
        # print("Findings: ", v['Findings'])
        # print("Checklist: ", v['Checklist'])
        try:
            tumor_diff.append(v['Checklist'][6])
            if 'well' in v['Checklist'][6]:
                tumor_diff_count['well'] += 1
            elif 'moderately' in v['Checklist'][6]:
                tumor_diff_count['moderately'] += 1
            elif 'poorly' in v['Checklist'][6]:
                tumor_diff_count['poorly'] += 1
                print("Report number: ", k)
                print("Findings: ", v['Findings'])
                print("Checklist: ", v['Checklist'])
        except:
            # print("Report number: ", k)
            # print("Findings: ", v['Findings'])
            # print("Checklist: ", v['Checklist'])
            continue
        # time.sleep(3)
    # print(tumor_diff)
    print(tumor_diff_count)
