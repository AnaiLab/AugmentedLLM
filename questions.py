from modules.MCQ import MCQ

# Note, questions are constructed in the following format:
# MCQ('Question Text', 'Correct Answer')

Questions = [
    MCQ("""What is 5^3? 
        A: 125 
        B: 625 
        C: 555
        D: 5""",'A'),

    MCQ("""What sound does a duck make? 
        A: Woof woof 
        B: Moo Moo
        C: Quack quack
        D: Oink Oink""",'C'),
    
    MCQ("""What is the standard first-line chemotherapy regimen for newly diagnosed advanced-stage epithelial ovarian cancer? 
        A) Bevacizumab and PARP inhibitors 
        B) Carboplatin and paclitaxel 
        C) Doxorubicin and cisplatin 
        D) Gemcitabine and docetaxel""",'B')
]