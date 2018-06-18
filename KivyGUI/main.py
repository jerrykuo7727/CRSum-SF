# -*- coding: utf-8 -*-
import os
import sys
import time
import dill
import pickle
import jieba
import numpy as np
import pandas as pd
from collections import Counter

import kivy.resources
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen, RiseInTransition
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.progressbar import ProgressBar
from kivy.properties import BooleanProperty
from kivy.uix.image import Image
from kivy.uix.checkbox import CheckBox
from kivy.clock import Clock
from kivy import Config

Config.set('graphics', 'width', '1280')
Config.set('graphics', 'height', '720')

global use_maxlen
use_maxlen = True

Builder.load_string("""
<RootScreenManager>:
    Screen:
        canvas.before:
            Color:
                rgba: (0,150/255,136/255,1)
            Rectangle:
                pos: self.pos
                size: self.size

        AnchorLayout:
            BoxLayout:
                orientation: 'vertical'
                padding: [0, 30, 0, 0]
                size_hint: None, None
                size: 640, 100

                ProgressBar:
                    id: loading_pb
                    size_hint_y: None
                    height: 20
                    max: 1000

                LoadingLabel:
                    id: loading_label
                    font_size: 28
                    size_hint_y: None
                    height: 50
                    on_rendered:
                        self.load_everything(loading_pb, root)

    Screen:
        name: 'main_screen'
        BoxLayout:
            orientation: 'vertical'
            padding: [80, 20, 80, 20]
            canvas.before:
                Color:
                    rgba: (0,91/255,78/255,1)
                Rectangle:
                    pos: self.pos
                    size: self.size

            OuterScreenManager:
                id: outer_sm
                Screen:
                    name: 'input_screen'
                    BoxLayout:
                        orientation: 'vertical'
                        BoxLayout:
                            size_hint_y: None
                            height: 100
                            AnchorLayout:
                                anchor_x: 'left'
                                StackLayout:
                                    BoxLayout:
                                        size_hint_x: None
                                        width: 185
                                        MyButton:
                                            id: docfc_btn
                                            background_color: 0,0,0,0 
                                            canvas.before:
                                                Color:
                                                    rgba: (.4,.4,.4,1) if self.state == 'normal' and not self.choosing else (0,150/255,136/255,1)
                                                RoundedRectangle:
                                                    pos: self.pos
                                                    size: self.size
                                                    radius: [30, 30, 0, 0]

                                            text: '讀取文章'
                                            font_size: 28
                                            size_hint_x: None
                                            width: 180
                                            border: 20,20,20,20
                                            on_press:
                                                self.user_press(titlefc_btn, inner_sm, 'document_file_chooser')

                                    MyButton:
                                        id: titlefc_btn
                                        background_color: 0,0,0,0 
                                        canvas.before:
                                            Color:
                                                rgba: (.4,.4,.4,1) if self.state == 'normal' and not self.choosing else (0,150/255,136/255,1)
                                            RoundedRectangle:
                                                pos: self.pos
                                                size: self.size
                                                radius: [30, 30, 0, 0]

                                        text: '讀取標題'
                                        font_size: 28
                                        size_hint_x: None
                                        width: 180
                                        on_press:
                                            self.user_press(docfc_btn, inner_sm, 'summary_file_chooser')

                            AnchorLayout:
                                anchor_x: 'right'
                                SettingBtn:
                                    on_release:
                                        self.settings_popup()
                                    background_color: 0,0,0,0 
                                    size_hint_x: None
                                    width: 180
                                    Image:
                                        source: 'gear.png'
                                        y: self.parent.y + self.parent.height - 90
                                        x: self.parent.x+45
                                        size: 80, 80
                                        allow_stretch: True

                        ScreenManager:
                            id: inner_sm

                            Screen:
                                name: 'text'
                                BoxLayout:
                                    orientation: 'vertical'
                                    
                                    ScrollView:
                                        id: scrlv_doc

                                        MyTextInput:
                                            id: document_input
                                            hint_text: '請輸入文章'
                                            font_size: 28
                                            size_hint_y: None
                                            height: max(self.minimum_height, scrlv_doc.height)
                                            padding_x: 20
                                            padding_y: 10

                                    BoxLayout:
                                        size_hint_y: None
                                        height: 3

                                    ScrollView:
                                        id: scrlv_title
                                        size_hint_y: None
                                        height: 120
                                        MyTextInput:
                                            id: summary_input
                                            hint_text: '(選填) 輸入文章標題以計算ROUGE-1/2'
                                            font_size: 28
                                            size_hint_y: None
                                            height: max(self.minimum_height, scrlv_title.height)
                                            padding_x: 20
                                            padding_y: 10

                            Screen:
                                name: 'document_file_chooser'
                                canvas.before:
                                    Color:
                                        rgba: (0,150/255,136/255,1)
                                    RoundedRectangle:
                                        pos: self.pos
                                        size: self.size
                                        radius: [0, 30, 30, 30]

                                MyFileChooser:
                                    show_hidden: True
                                    filters: ['*.txt']
                                    on_submit:
                                        docfc_btn.choosing = False
                                        docfc_btn.state = 'down'
                                        docfc_btn.state = 'normal'
                                        document_input.text = self.read_file(args[1][0])
                                        inner_sm.transition.direction = 'up'
                                        inner_sm.current = 'text'

                            Screen:
                                name: 'summary_file_chooser'
                                canvas.before:
                                    Color:
                                        rgba: (0,150/255,136/255,1)
                                    RoundedRectangle:
                                        pos: self.pos
                                        size: self.size
                                        radius: [0, 30, 30, 30]

                                MyFileChooser:
                                    show_hidden: True
                                    filters: ['*.txt']
                                    on_submit:
                                        titlefc_btn.choosing = False
                                        titlefc_btn.state = 'down'
                                        titlefc_btn.state = 'normal'
                                        summary_input.text = self.read_file(args[1][0])
                                        inner_sm.transition.direction = 'up'
                                        inner_sm.current = 'text'

                Screen:
                    name: 'output_screen'
                    BoxLayout:
                        orientation: 'vertical'

                        BoxLayout:
                            id: title_layout
                            orientation: 'vertical'
                            size_hint_y: None
                            height: 150
                            opacity: 1
                            MyLabel:
                                size_hint: None, None
                                height: 50
                                text: '標題：'
                                font_size: 28

                            ScrollView:
                                size_hint_y: None
                                height: 100
                                canvas.before:
                                    Color:
                                        rgba: (224/255,242/255,241/255,1)
                                    RoundedRectangle:
                                        pos: self.pos
                                        size: self.size
                                        radius: [10, 10, 10, 10]

                                MyLabel:
                                    id: title_output
                                    font_size: 28
                                    color: 0,0,0,1
                                    size_hint_y: None
                                    text_size: self.width, None
                                    height: self.texture_size[1]
                                    padding_x: 20
                                    padding_y: 5

                        MyLabel:
                            size_hint: None, None
                            height: 50
                            text: '摘要：'
                            font_size: 28

                        ScrollView:
                            size_hint_y: None
                            height: 100
                            canvas.before:
                                Color:
                                    rgba: (224/255,242/255,241/255,1)
                                RoundedRectangle:
                                    pos: self.pos
                                    size: self.size
                                    radius: [10, 10, 10, 10]

                            MyLabel:
                                id: summary_output
                                font_size: 28
                                color: 0,0,0,1
                                text_size: self.width, None
                                size_hint_y: None
                                height: self.texture_size[1]
                                padding_x: 20
                                padding_y: 5

                        MyLabel:
                            size_hint: None, None
                            height: 50
                            text: '內文：'
                            font_size: 28

                        ScrollView:
                            canvas.before:
                                Color:
                                    rgba: (224/255,242/255,241/255,1)
                                RoundedRectangle:
                                    pos: self.pos
                                    size: self.size
                                    radius: [10, 10, 10, 10]
                            MyLabel:
                                id: document_output
                                font_size: 28
                                color: 0,0,0,1
                                text_size: self.width, None
                                size_hint_y: None
                                height: self.texture_size[1]
                                padding_x: 20
                                padding_y: 5

            BoxLayout:
                size_hint_y: None
                height: 100

                AnchorLayout:
                    anchor_x: 'left'
                    BoxLayout:
                        orientation: 'vertical'
                        MyLabel:
                            id: rouge_score_r
                            opacity: 0
                            text: 'ROUGE-1/2：34.27 / 10.60'
                            size_hint_x: None
                            width: 400
                            font_size: 28
                        MyLabel:
                            id: rouge_score_p
                            opacity: 0
                            text: 'ROUGE-1/2：34.27 / 10.60'
                            size_hint_x: None
                            width: 400
                            font_size: 28
                        MyLabel:
                            id: rouge_score_f
                            opacity: 0
                            text: 'ROUGE-1/2：34.27 / 10.60'
                            size_hint_x: None
                            width: 400
                            font_size: 28

                AnchorLayout:
                    padding: [0, 10, 120, 0]
                    anchor_x: 'right'
                    GenerateBtn:
                        background_color: 0,0,0,0 
                        canvas.before:
                            Color:
                                rgba: (.4,.4,.4,1) if self.state == 'normal' else (0,150/255,136/255,1)
                            RoundedRectangle:
                                pos: self.pos
                                size: self.size
                                radius: [30, 30, 30, 30]

                        text: '生成摘要'
                        font_size: 28
                        size_hint_x: None
                        width: 400
                        on_release:
                            outer_sm.switch_screen(self, document_input, summary_input, document_output, \
                                                summary_output, title_output, title_layout, rouge_score_r, \
                                                rouge_score_p, rouge_score_f, inner_sm, docfc_btn, titlefc_btn)
""")

###################
#    資料預處理    #
###################

def contains_header_words(string):
    header_words = ["中國時報", "工商時報", "旺報", "▲"]
    contain = False
    for word in header_words:
        contain = contain or word in string
    return contain

def parse_to_sentences(document):
    try:
        buffer, sentences = "", []
        header_tag, skip_flag = False, False

        for i, char in enumerate(document):
            if skip_flag:
                skip_flag = False
                continue

            buffer += ' ' if char == '\n' else char

            # 檢查文章開頭是否為特別標籤
            if not sentences:
                if char in "(（【":
                    if not buffer[:-1].strip() or contains_header_words(buffer):
                        header_tag = True
                        continue

                if not header_tag and char == '導' and '／' in buffer and '報導' in buffer:
                    sentence = buffer.strip()
                    if sentence:
                        sentences.append(sentence)
                        buffer = ""

            if header_tag:
                if char in "】）)":
                    sentence = buffer.strip()
                    sentences.append(sentence)
                    buffer, header_tag = "", False
                    continue

            # 基於標點符號作分句        
            if char in '，。?？!！:：;；':
                try:
                    if document[i+1] == "」":
                        buffer += document[i+1]
                        skip_flag = True
                except: None

                sentence = buffer.strip()
                if sentence:
                    sentences.append(sentence)
                    buffer = ""

        sentence = buffer.strip()
        if sentence:
            sentences.append(sentence)
        
        return sentences if sentences else None
    except:
        return None

def get_sen_labels():
    sen_labels = []
    for i in reversed(range(5)):
        sen_labels.append('stm'+str(i+1))
    sen_labels.append('st')
    for i in range(5):
        sen_labels.append('stn'+str(i+1))
    return sen_labels

def padding(sentence):
    sentence.insert(0, '<L>')
    sentence.append('<R>')
    return sentence

def get_tf(row, word2tf):
    tf = 0
    for word in row.st:
        tf += word2tf[word]
    return tf / len(row.st)

def get_df(row, word2df):
    df = 0
    for word in row.st:
        df += word2df[word]
    return df / len(row.st)

def parse_to_df(document, word2df, len_scaler):
    document = [list(jieba.cut(sen)) for sen in parse_to_sentences(document)]
    flatten_doc = [word for sen in document for word in sen]
    doc_len = len(flatten_doc)

    word2tf = Counter(flatten_doc)
    for word in word2tf:
        word2tf[word] /= doc_len

    word_cursor, pos_list = 0, []
    for i, sentence in enumerate(document):
        word_cursor += len(sentence)
        pos_list.append(word_cursor / doc_len)

    sen_list, sentences_list = [], []
    document = list(map(padding, document))
    for _ in range(5):
        document.insert(0, [])
    for _ in range(5):
        document.append([])
    
    for i in range(len(document)-10):
        sen_list.append(i)
        sentences_list.append(document[i:i+11])

    sen_df = pd.DataFrame(sen_list)
    sentences_df = pd.DataFrame(sentences_list)
    pos_df = pd.DataFrame(pos_list)

    sentence_df = pd.concat([sen_df, sentences_df], axis=1)
    sentence_df.columns = ['sen'] + get_sen_labels()

    len_df = sentence_df.st.apply(lambda x: len(x)-2)
    tf_df = sentence_df.apply(lambda x: get_tf(x, word2tf), axis=1)
    df_df = sentence_df.apply(lambda x: get_tf(x, word2df), axis=1)

    df = pd.concat([sentence_df, len_df, pos_df, tf_df, df_df], axis=1)   
    df.columns = ['sen'] + get_sen_labels() + ['len', 'pos', 'tf', 'df']
    df.len = len_scaler.transform(df.len.values.reshape(-1, 1))
    return df

##########################
#    Sentence Scoring    #
##########################

def tokenizing(text, word2token):
    return [word2token[word] for word in text]

def process_text(text, pad_len, word2token):
    return pad_sequences(text.apply(lambda x: tokenizing(x, word2token)),
                         maxlen=pad_len, padding='post', truncating='post')

def predict_rouge_2(df, pad_len, word2token, model):
    inputs = {}
    for label in get_sen_labels():
        inputs[label] = process_text(df[label], pad_len, word2token)
    inputs['sf'] = np.array(df[['len', 'pos', 'tf', 'df']])
    return model.predict(inputs)

############################
#    Sentence Selection    #
############################

def length_of(df):
    try:
        return df.st.apply(lambda x: len(x[1:-1])).sum()
    except:
        return 0

def get_bigrams(sen):
    bigram = set()
    try:
        for i in range(0, len(sen)-1):
            bigram.add((sen[i], sen[i+1]))
    except: None
    return bigram
    

def bigram_overlap(st_df, psi_df):
    try:
        st = [word for st in st_df.st.values for word in st[1:-1]]
        psi = [word for st in psi_df.st.values for word in st[1:-1]]
        st_bigrams = get_bigrams(st)
        psi_bigrams = get_bigrams(psi)

        overlap = 0
        for bigram in st_bigrams:
            overlap += bigram in psi_bigrams
        return overlap / len(st)
    except:
        return 0

def get_summary(df, rouge_2_label='rouge_2', maxsen=None, maxlen=33, redundancy=0.3, ordered=1):
    rouge2 = df[rouge_2_label].values
    
    psi_df = pd.DataFrame()
    if not maxsen:
        while np.max(rouge2) != -1 and length_of(psi_df) < maxlen:
            sen = np.argmax(rouge2)
            rouge2[sen] = -1
            
            sen_df = df[df.sen == sen][['sen', 'st']]
            if length_of(psi_df) + length_of(sen_df) > maxlen:
                continue
            if bigram_overlap(sen_df, psi_df) <= redundancy:
                psi_df = pd.concat((psi_df, sen_df))
    else:
        maxsen = int(round(len(df)*0.1))
        maxsen = 1 if maxsen == 0 else maxsen
        while np.max(rouge2) != -1 and len(psi_df) < maxsen:
            sen = np.argmax(rouge2)
            rouge2[sen] = -1
            
            sen_df = df[df.sen == sen][['sen', 'st']]
            if bigram_overlap(sen_df, psi_df) <= redundancy:
                psi_df = pd.concat((psi_df, sen_df))
    
    try:
        if ordered:
            psi_df = psi_df.sort_values('sen')
        return [word for st in psi_df.st.values for word in st[1:-1]]
    except:
        return []
    
    try:
        if ordered:
            psi_df = psi_df.sort_values('sen')
        return [word for st in psi_df.st.values for word in st[1:-1]]
    except:
        return []
        
def rouge_1_2(title_list, summary_list):
    title_bigrams = get_bigrams(title_list)
    summary_bigrams = get_bigrams(summary_list)
    print(title_list)
    print(summary_list)

    try:
        unigram_overlap = 0
        for word in set(summary_list):
            unigram_overlap += word in set(title_list)
        rouge_1_r = unigram_overlap / len(set(title_list)) * 100
    except:
        rouge_1_r = 0

    try:
        bigram_overlap = 0
        for bigram in summary_bigrams:
            bigram_overlap += bigram in title_bigrams
        rouge_2_r = bigram_overlap / len(title_bigrams) *100
    except:
        rouge_2_r = 0

    try:
        unigram_overlap = 0
        for word in set(title_list):
            unigram_overlap += word in set(summary_list)
        rouge_1_p = unigram_overlap / len(set(summary_list)) * 100
    except:
        rouge_1_p = 0

    try:
        bigram_overlap = 0
        for bigram in title_bigrams:
            bigram_overlap += bigram in summary_bigrams
        rouge_2_p = bigram_overlap / len(summary_bigrams) *100
    except:
        rouge_2_p = 0
    
    try:
        rouge_1_f = 2 * rouge_1_p * rouge_1_r / (rouge_1_p + rouge_1_r)
    except:
        rouge_1_f = 0
    
    try:
        rouge_2_f = 2 * rouge_2_p * rouge_2_r / (rouge_2_p + rouge_2_r)
    except:
        rouge_2_f = 0
        
    return rouge_1_r, rouge_2_r, rouge_1_p, rouge_2_p, rouge_1_f, rouge_2_f



def load_helpers(self, pb, label, sm):
    global len_scaler, pad_len, word2df, word2token
    fn_len_scaler = getResourcePath('len_scaler.pkl')
    fn_pad_len = getResourcePath('pad_len.pkl')
    fn_word2df = getResourcePath('word2df.dill')
    fn_word2token = getResourcePath('word2token.dill')

    with open(fn_len_scaler, 'rb') as file:
        len_scaler = pickle.load(file)
    with open(fn_pad_len, 'rb') as file:
        pad_len = pickle.load(file)
    with open(fn_word2df, 'rb') as file:
        word2df = dill.load(file)
    with open(fn_word2token, 'rb') as file:
        word2token = dill.load(file)
    use_maxlen = True

    pb.value = 200
    label.text = "正在加載函式庫..."
    Clock.schedule_once(lambda x: load_modules(x, pb, label, sm), 0)

def load_modules(self, pb, label, sm):
    global pad_sequences, load_model
    from keras.preprocessing.sequence import pad_sequences
    from keras.models import load_model
    jieba.set_dictionary(getResourcePath('dict.txt.big'))
    jieba.initialize()

    pb.value = 600
    label.text = "正在加載CRSum+SF模型 (請稍候...)"
    Clock.schedule_once(lambda x: load_models(x, pb, label, sm), 0)

def load_models(self, pb, label, sm):
    global crsum_sf_model
    fn_crsum_sf = getResourcePath('crsum_sf.h5')
    crsum_sf_model = load_model(fn_crsum_sf)

    pb.value = 1000
    label.text = "完成加載！"
    Clock.schedule_once(lambda x: load_main(x, sm), 0)

def load_main(self, sm):
    time.sleep(0.6)
    sm.current = 'main_screen'
    
    
class RootScreenManager(ScreenManager):
    pass

class LoadingLabel(Label):
    rendered = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(LoadingLabel, self).__init__(**kwargs)
        fontpath = 'DroidSansFallback.ttf'
        if '_MEIPASS2' in os.environ:
            fontpath = os.path.join(os.environ['_MEIPASS2'], fontpath)
        self.font_name = fontpath
        Clock.schedule_once(self.render, 0)

    def render(self, *args):
        self.rendered = True

    def load_everything(self, pb, sm):
        global start_time
        start_time = time.time()
        self.text = "正在加載Helpers..."
        Clock.schedule_once(lambda x: load_helpers(x, pb, self, sm), 0)

class MyButton(Button):
    def __init__(self, **kwargs):
        super(MyButton, self).__init__(**kwargs)
        fontpath = 'DroidSansFallback.ttf'
        if '_MEIPASS2' in os.environ:
            fontpath = os.path.join(os.environ['_MEIPASS2'], fontpath)
        self.font_name = fontpath
        self.choosing = False

    def user_press(self, other_btn, sm, sm_current=None):
        if not self.choosing:
            self.choosing = True
            other_btn.choosing = False
            other_btn.state = 'down'
            other_btn.state = 'normal'
            sm.transition.direction = 'down'
            sm.current = sm_current
        else:
            self.choosing = False
            self.state = 'down'
            self.state = 'normal'
            sm.transition.direction = 'up'
            sm.current = 'text'

class MyTextInput(TextInput):
    def __init__(self, **kwargs):
        super(MyTextInput, self).__init__(**kwargs)
        fontpath = 'DroidSansFallback.ttf'
        if '_MEIPASS2' in os.environ:
            fontpath = os.path.join(os.environ['_MEIPASS2'], fontpath)
        self.font_name = fontpath

class MyLabel(Label):
    def __init__(self, **kwargs):
        super(MyLabel, self).__init__(**kwargs)
        fontpath = 'DroidSansFallback.ttf'
        if '_MEIPASS2' in os.environ:
            fontpath = os.path.join(os.environ['_MEIPASS2'], fontpath)
        self.font_name = fontpath

class SettingBtn(Button):
    def settings_popup(self):
        global use_maxlen
        fontpath = 'DroidSansFallback.ttf'
        if '_MEIPASS2' in os.environ:
            fontpath = os.path.join(os.environ['_MEIPASS2'], fontpath)

        content = BoxLayout(orientation='vertical')
        option1 = BoxLayout(padding=[20, 0, 20, 0])
        option2 = BoxLayout(padding=[20, 0, 20, 0])

        radio_btn1 = CheckBox(group= "gr", active=use_maxlen, size_hint_x=None, width=30)
        radio_btn1.bind(active=self.change_settings)
        option1.add_widget(radio_btn1)
        option1.add_widget(Label(text="選出最多30個詞", font_name=fontpath, font_size=28))
        option2.add_widget(CheckBox(group= "gr", active=(not use_maxlen), size_hint_x=None, width=30))
        option2.add_widget(Label(text="選出總句數的10%", font_name=fontpath, font_size=28))
        dismiss_btn = Button(size_hint_y=None, height=50, text='完成', font_name=fontpath, font_size=28)

        content.add_widget(option1)
        content.add_widget(option2)
        content.add_widget(dismiss_btn)
        
        popup = Popup(title='設定', content=content, size_hint=(None, None), size=(330, 300), title_font=fontpath, title_size=28)
        dismiss_btn.bind(on_press=popup.dismiss)
        popup.open()
    
    def change_settings(self, instance, value):
        global use_maxlen
        use_maxlen = value

class OuterScreenManager(ScreenManager):
    def warn_bad_input(self):
        fontpath = 'DroidSansFallback.ttf'
        if '_MEIPASS2' in os.environ:
            fontpath = os.path.join(os.environ['_MEIPASS2'], fontpath)

        content = BoxLayout(orientation='vertical')
        content.add_widget(Label(text="您尚未輸入文章！", font_name=fontpath, font_size=28))
        dismiss_btn = Button(size_hint_y=None, height=50, text='知道了', font_name=fontpath, font_size=28)
        content.add_widget(dismiss_btn)
        popup = Popup(title='錯誤', content=content, size_hint=(None, None), size=(640, 360), title_font=fontpath, title_size=28)
        dismiss_btn.bind(on_press=popup.dismiss)
        popup.open()

    def switch_screen(self, gen_btn, document_input, summary_input, document_output, 
                      summary_output, title_output, title_layout, rouge_score_r,
                      rouge_score_p, rouge_score_f, inner_sm, docfc_btn, titlefc_btn):
        if not document_input.text.strip():
            self.warn_bad_input()
            return
        
        if self.current == 'input_screen':
            df = parse_to_df(document_input.text, word2df, len_scaler)
            df['rouge_2'] = predict_rouge_2(df, pad_len, word2token, crsum_sf_model)
            summary_list = get_summary(df) if use_maxlen else get_summary(df, maxsen=True)
            
            if summary_input.text.strip():
                title_list = list(jieba.cut(summary_input.text))
                title_layout.height = 150
                title_layout.opacity = 1
                rouge_1_r, rouge_2_r, rouge_1_p, rouge_2_p, rouge_1_f, rouge_2_f = rouge_1_2(title_list, summary_list)
                #rouge_score_r.text = "ROUGE-1/2 (recall)：%.2f / %.2f" % (rouge_1_r, rouge_2_r)
                #rouge_score_p.text = "ROUGE-1/2 (precision)：%.2f / %.2f" % (rouge_1_p, rouge_2_p)
                #rouge_score_f.text = "ROUGE-1/2 (f-score)：%.2f / %.2f" % (rouge_1_f, rouge_2_f)
                rouge_score_p.text = "ROUGE-1/2：%.2f / %.2f" % (rouge_1_r, rouge_2_r)
                #rouge_score_r.opacity = 1
                rouge_score_p.opacity = 1
                #rouge_score_f.opacity = 1
            else:
                title_layout.height = 0
                title_layout.opacity = 0
                rouge_score_r.opacity = 0
                rouge_score_p.opacity = 0
                rouge_score_f.opacity = 0

            title_output.text = summary_input.text
            summary_output.text = ''.join(summary_list)
            document_output.text = document_input.text

            self.transition.direction = 'left'
            self.current = 'output_screen'
        else:
            document_input.text = ""
            summary_input.text = ""
            self.transition.direction = 'right'
            self.current = 'input_screen'

        gen_btn.switch_text(rouge_score_r, rouge_score_p, rouge_score_f, inner_sm, docfc_btn, titlefc_btn)

class GenerateBtn(Button):
    def __init__(self, **kwargs):
        super(GenerateBtn, self).__init__(**kwargs)
        fontpath = 'DroidSansFallback.ttf'
        if '_MEIPASS2' in os.environ:
            fontpath = os.path.join(os.environ['_MEIPASS2'], fontpath)
        self.font_name = fontpath

    def switch_text(self, rouge_score_r, rouge_score_p, rouge_score_f, inner_sm, docfc_btn, titlefc_btn):
        if self.text == '生成摘要':
            self.text = '回到上一頁'
        else:
            rouge_score_r.opacity = 0
            rouge_score_p.opacity = 0
            rouge_score_f.opacity = 0
            docfc_btn.choosing = False
            docfc_btn.state = 'down'
            docfc_btn.state = 'normal'
            titlefc_btn.choosing = False
            titlefc_btn.state = 'down'
            titlefc_btn.state = 'normal'
            inner_sm.current = 'text'
            self.text = '生成摘要'

class MyFileChooser(FileChooserIconView):
    def __init__(self, **kwargs):
        super(MyFileChooser, self).__init__(**kwargs)
        self.path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    def read_file(self, filepath):
        try:
            with open(filepath, mode='r', encoding='utf-8') as file:
                return file.read()
        except:
            try:
                with open(filepath, mode='r', encoding='ansi') as file:
                    return file.read()
            except:
                fontpath = 'DroidSansFallback.ttf'
                if '_MEIPASS2' in os.environ:
                    fontpath = os.path.join(os.environ['_MEIPASS2'], fontpath)  

                content = BoxLayout(orientation='vertical')
                content.add_widget(Label(text="文件編碼必須是UTF-8或ANSI！", font_name=fontpath, font_size=28))
                dismiss_btn = Button(size_hint_y=None, height=50, text='知道了', font_name=fontpath, font_size=28)
                content.add_widget(dismiss_btn)
                popup = Popup(title='錯誤', content=content, size_hint=(None, None), size=(640, 360), title_font=fontpath, title_size=28)
                dismiss_btn.bind(on_press=popup.dismiss)
                popup.open()
                return ""

class MyApp(App):
    def build(self):
        self.title = 'CRSum抽取式摘要系統'
        self.icon = getResourcePath('icon.png')
        return RootScreenManager(transition=RiseInTransition(duration=0.4))

def getResourcePath(filename):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.abspath("."), filename)

def resourcePath():
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS)
    return os.path.join(os.path.abspath("."))

if __name__ == '__main__':
    kivy.resources.resource_add_path(resourcePath()) # add this line
    MyApp().run()