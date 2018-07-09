import xlrd
import numpy as np
import xlwt
from xlutils.copy import copy

workbook_origin_path = u'D:\MyConfiguration\szj46941\Desktop\(专业知识点-实例导出)(20180704145958).xls'
workbook_formal_question_path = u'D:\MyConfiguration\szj46941\Desktop\导入模板.xlsx'
workbook_extension_question_path = u'D:\MyConfiguration\szj46941\Desktop\语料上传.xlsx'
workbook_test_question_path = u'D:\MyConfiguration\szj46941\Desktop\测试题上传.xlsx'
count = 0

def get_sheet(sheet_index):
    workbook = xlrd.open_workbook(workbook_origin_path)
    sheet_name = workbook.sheet_names()[sheet_index]
    return workbook.sheet_by_name(sheet_name)


def get_question_row_start_indexes(sheet):
    cols = np.array(sheet.col_values(13))
    return np.where(cols != '')[0][1:]


def write_formal(q_list, a_list):
    workbook = xlrd.open_workbook(workbook_formal_question_path)
    workbooknew = copy(workbook)
    ws = workbooknew.get_sheet(0)
    for i in range(len(q_list)):
        question = q_list[i]
        answer = a_list[i]
        ws.write(i + 1, 1, question)
        ws.write(i + 1, 2, '知识')
        ws.write(i + 1, 3, answer)
        ws.write(i + 1, 5, '一般')
        ws.write(i + 1, 6, '2018/06/05 00:00:00')
        ws.write(i + 1, 7, '2028/06/05 00:00:00')

    workbooknew.save(u'formal.xlsx')


def write_extension(q_list, s_list):
    workbook = xlrd.open_workbook(workbook_extension_question_path)
    workbooknew = copy(workbook)
    ws = workbooknew.get_sheet(0)
    row_index = 1
    for i in range(len(q_list)):
        question = q_list[i]
        extensions = s_list[i][:int(len(s_list[i]) * 9 / 10)]
        for q in extensions:
            ws.write(row_index, 0, q)
            ws.write(row_index, 1, question)
            row_index += 1
    workbooknew.save(u'extension.xlsx')


def write_test(q_list, s_list):
    workbook = xlrd.open_workbook(workbook_test_question_path)
    workbooknew = copy(workbook)
    ws = workbooknew.get_sheet(0)
    row_index = 1
    for i in range(len(q_list)):
        question = q_list[i]
        extensions = s_list[i][int(len(s_list[i]) * 9 / 10):]
        if len(extensions) == 0:
            extensions = s_list[i][0]
        for q in extensions:
            ws.write(row_index, 0, q)
            ws.write(row_index, 1, question)
            row_index += 1
    workbooknew.save(u'test.xlsx')


if __name__ == '__main__':
    sheet = get_sheet(0)
    q_row_start_indexes = get_question_row_start_indexes(sheet)

    q_list = []
    a_list = []
    s_list = []

    for i in range(len(q_row_start_indexes)):
        count += 1
        if i != len(q_row_start_indexes) - 1:
            question = ''
            answer = ''
            similar_questions = []
            q_start = q_row_start_indexes[i]
            q_end = q_row_start_indexes[i + 1]
            for j in range(q_start, q_end):
                rows = sheet.row_values(j)
                if j == q_start:
                    question = str.strip(rows[13])
                    answer = str.strip(rows[18])
                else:
                    if rows[16] != '':
                        similar_questions.append(str.strip(rows[16]))
            if len(similar_questions) >= 30:
                q_list.append(question)
                a_list.append(answer)
                s_list.append(similar_questions)

    write_formal(q_list, a_list)
    write_extension(q_list, s_list)
    write_test(q_list, s_list)
    print(count)
