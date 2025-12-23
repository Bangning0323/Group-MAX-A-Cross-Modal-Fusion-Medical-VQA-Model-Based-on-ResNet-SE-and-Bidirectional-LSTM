class QuestionTypeClassifier:

    def __init__(self):
        # Yes/No
        self.yesno_keywords = {
            'is', 'are', 'was', 'were', 'do', 'does', 'did', 'will', 'would',
            'can', 'could', 'may', 'might', 'should', 'shall', 'has', 'have', 'had',
            'is there', 'are there', 'do you', 'does it', 'did they'
        }
        # Keywords for open-ended questions
        self.open_keywords = {
            'what', 'why', 'how', 'where', 'when', 'who', 'which', 'whose',
            'what is', 'what are', 'how many', 'how much', 'how long', 'where is',
            'why is', 'when is', 'which one', 'what type', 'what kind'
        }

    def classify(self, question):
        #Determine the type of question
        question_lower = question.strip().lower()
        for keyword in sorted(self.yesno_keywords, key=len, reverse=True):
            if question_lower.startswith(keyword):
                return 'yesno'
        for keyword in sorted(self.open_keywords, key=len, reverse=True):
            if question_lower.startswith(keyword):
                return 'open'

        return 'open'