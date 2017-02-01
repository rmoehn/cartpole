# Credits: http://stackoverflow.com/a/19829714/5091738
class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value
