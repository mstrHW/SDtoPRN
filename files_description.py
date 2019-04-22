eho_description = {
    'id_columns': ['id'],
    'float_columns': [
        'Mfu ФУ',
        'Mfvs ФВ симпсона ',
        'Mfvt ФВ Тейхольц ',
        'Mimm ИММ',
        'Mkdo КДО',
        'Mkdr КДР',
        'Mkso КСО',
        'Mksr КСР',
        'Mmgp МПЖ',
        'Mmm ММ',
        'Muo УО',
        'Mzs ЗС',
        'Maorta аорта ',
        'Maortav  восходящая аорта ',
        'MLp  левое предсердие  ',
        'M4kam',
        'Mparp  Парастернальная позиция ',
        'Mpst передняя стенка   ',
        'Mprpr  правое предсердие ',
        'Mnpvd нижняя полая вена ',
        'Mnpvs спадение на входе ',
        'MakVmax Аортальный клапан Vmax ',
        'MakdPmax Аортальный клапан Pmax',
        'Meem Митральный  клапан E/ Em',
        'Mveva Митральный клапан Ve/Va ',
        'mkVa митральный клапан Va',
        'mkVe митральный клапан  Ve',
        'Mtkvd  трикуспидальный  клапан dPmax',
        'Mtkv трикуспидальный  клапан Vmax',
        'MpkVmax  пульмональный  клапан Vmax',
        'Mpkd пульмональный  клапан Vmax',
        'Mcss частота СС',
        'mkTdec',
    ],
    'categorical_columns': [
        'Mkin Кинетика ',
        'Mrit ритм ',
        'Mst состояние аорты ',
        'MLpo МПП ',
        'Makstv Аортальный клапан створки ',
        'Makregur Аортальный клапан регургитация ',
        'mkregur  митральный клапан  регургитация ',
        'mkstv митральный клапан створки',
        'Mtkregur трикуспидальный  клапан  регургитация ',
        'Mtkstv  трикуспидальный  клапан створки',
        'Mpkreg пульмональный  клапан  регургитация ',
    ],
    'date_columns': ['date'],
    'unused_columns': [
        'gap',
        'MtkdPtr'
    ],
}


events_description = {
    'id_columns': [
        'patient_id',
        'epizod_id',
    ],
    'float_columns': [
    ],
    'categorical_columns': [
    ],
    'date_columns': [
        'data'
    ],
    'unused_columns': [
        'event_id',
        'cod',
        'department'
    ],
}


holesterin_description = {
    'id_columns': [
        'pacient_id -',
        'epizod_id -',
    ],
    'float_columns': [
        'age',
        'durationcalc ',
        'MIN_HGB',
        'MAX_Troponin',
        'MAX_ALT',
        'MAX_ACT',
        'MAX_Creatinine',
        'MAX_Glucouse',
        'MAX_Leukocytes',
        'MAX_PLT',
        'holesterin',
    ],
    'categorical_columns': [
        'ibs',
        'gb',
        'stenokard',
        'q',
        'hbp',
        'hobl',
        'xcn',
        'oasnk',
        'sd',
        'sex',
        'исход  -',
        'состояние при поступлении ',
        'I20',
        'I21',
        'I22',
    ],
    'date_columns': [
        'datep -',
        'enddate -',
    ],
    'unused_columns': [
        'result -',
        'porgosp',
        'diagstac',
        'placeper -',
        'DurationReanFact -',
        'DurationStacionar -',
        'anesthesia',
        'stent_name1',
        'stent_size1',
        'stent_pressure1',
        'stent_place1',
        'stent_name2',
        'stent_size2',
        'stent_pressure2',
        'stent_place2',
        'stent_name3',
        'stent_size3',
        'stent_pressure3',
        'stent_place3 -',
        'sdodscache -',
        'экстренность ',
        'повторность ',
        'результат -',
        'dateendcache -',
        'dlitday - цель ',
        'dclindiagcache -',
        'CaseID - ',
    ],
}


labresult_description = {
    'id_columns': [
        'patient_id',
        'epizod_id',
        'cod1',
        'cod2',
    ],
    'float_columns': [
        'Troponin',
        'RBC',
        'WBC',
        'HGB',
        'HCT',
        'PLT',
        'AST',
        'ALT',
        'Kreatinin',
        'Glucose',
        'Holesterin',
    ],
    'categorical_columns': [
        'result',
        'condition',
    ],
    'date_columns': [
        'start_date',
        'end_date',
        'event_date',
    ],
    'unused_columns': [
        'Department',
        'gap',
    ],
}


predst_description = {
    'id_columns': [
        'id',
    ],
    'float_columns': [
        'рост',
        ' Вес',
        'индекс массы тела',
        'площадь',
        'pressure',
        'Температура_тела',
        'Температура_тела_вечерняя',
    ],
    'categorical_columns': [
        'condition',
    ],
    'date_columns': [
        'date',
    ],
    'unused_columns': [
    ],
}
