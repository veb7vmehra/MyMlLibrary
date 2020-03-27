from kivy.app import App
from kivy.uix.button import  Button
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.core.window import Window
import random
  
class BreastCancerApp(App):
 
    def build(self):
        Clock.schedule_interval(self.Rainbow,0.2)
        self.box = BoxLayout(orientation='vertical', spacing=5,size_hint=(0.6,0.95),pos=(Window.width/8,Window.height/16))

        self.titl = Label(text='Breast Cancer Type Predictor',font_size='40')
        self.txt1 = TextInput(hint_text='Clump Thickness', size_hint=(1.2,.4))
        self.txt2 = TextInput(hint_text='Uniformity of Cell Size', size_hint=(1.2,.4))
        self.txt3 = TextInput(hint_text='Uniformity of cell shape', size_hint=(1.2,.4))
        self.txt4 = TextInput(hint_text='Marginal Adhesion', size_hint=(1.2,.4))
        self.txt5 = TextInput(hint_text='Single Apethelial Cell Size', size_hint=(1.2,.4))
        self.txt6 = TextInput(hint_text='Bare Neuclie', size_hint=(1.2,.4))
        self.txt7 = TextInput(hint_text='Bland Chromatin', size_hint=(1.2,.4))
        self.txt8 = TextInput(hint_text='Normal Neuclioli', size_hint=(1.2,.4))
        self.txt9 = TextInput(hint_text='Mitoses', size_hint=(1.2,.4))
        self.btn = Button(text='Predict Type of Cancer', on_press=self.AddEm, size_hint=(.7,.5),background_color=(0,255,0,1))
        self.smolbox = BoxLayout(orientation='horizontal',spacing=15)
        self.clear = Button(text='Reset', on_press=self.clearEm, size_hint=(.7,.5),background_color=(255,0,0,1))

        self.out = Label(text='', size_hint=(1.2,.1))

        self.box.add_widget(self.titl)
        self.box.add_widget(self.txt1)
        self.box.add_widget(self.txt2)
        self.box.add_widget(self.txt3)
        self.box.add_widget(self.txt4)
        self.box.add_widget(self.txt5)
        self.box.add_widget(self.txt6)
        self.box.add_widget(self.txt7)
        self.box.add_widget(self.txt8)
        self.box.add_widget(self.txt9)

        self.smolbox.add_widget(self.btn)

        self.smolbox.add_widget(self.clear)
 
        self.box.add_widget(self.smolbox)

        self.box.add_widget(self.out)

        return self.box
 
    def Rainbow(self, dt):
        Window.clearcolor = (random.randrange(0,255,1)/255, random.randrange(0,255,1)/255, random.randrange(0,255,1)/255, random.randrange(0,255,1)/255)
    def clearEm(self,instance):
        self.out.text=''
        self.txt1.text=''
        self.txt2.text=''
        self.txt3.text=''
        self.txt4.text=''
        self.txt5.text=''
        self.txt6.text=''
        self.txt7.text=''
        self.txt8.text=''
        self.txt9.text=''

    def AddEm(self, instance):

        arr = [int(self.txt1.text),int(self.txt2.text),int(self.txt3.text),int(self.txt4.text),int(self.txt5.text),int(self.txt6.text),int(self.txt7.text),int(self.txt8.text),int(self.txt9.text)]
        w1 =  [1.66256216,0.66636479,-0.78858303,-0.32599137,1.44862988,-0.80982349,-1.32509989,0.04951132,1.92897642]
        b1 = -0.41989057
        w2 = 3.39191686
        b2 = -2.48183147
        temp = 0

        for i in range(9):
            temp = arr[i]*w1[i]

        z1 = temp + b1
        z2 = w2*z1 + b2
        if(z2 < 3):
            self.out.text=str("Benign Breast Cancer")
        elif(z2 > 3):
            self.out.text=str("Melegnant Breast Cancer")

BreastCancerApp().run() 
