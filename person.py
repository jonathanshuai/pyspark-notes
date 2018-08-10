class Person:
    def parse(self, line):
        fields = line.split('\t')

        self.name = fields[0] 
        self.gender = fields[1] 
        self.age = fields[2] 
        self.language = fields[3] 

        return self

    def __repr__(self):
        return "<Name: {}, Gender: {}, Age: {}, Language: {}>"\
            .format(self.name, self.gender, self.age, self.language)


    def __str__(self):
        return ("Name: {}\nGender: {}\nAge: {}\nLanguage: {}"\
            .format(self.name, self.gender, self.age, self.language))