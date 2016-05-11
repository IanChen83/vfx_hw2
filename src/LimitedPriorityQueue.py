class LimitedPriorityQueue:
    def __init__(self, limit=0):
        self.__limit__ = limit
        self.queue = []

    def push_list(self, l):
        for x in l:
            self.push(x)

    def __str__(self):
        return "[\n\t" + '\n\t'.join([str(x) for x in self.queue]) + "\n]"

    def push(self, obj):
        if len(self.queue) == 0:
            self.queue.append(obj)
            return True

        if obj > self.queue[0]:
            return False

        i = 0
        length = len(self.queue)
        while i < length:
            if obj > self.queue[i]:
                self.queue.insert(i, obj)
                break
            else:
                i += 1

        if i == length:
            self.queue.append(obj)

        if self.__limit__ > 0 and len(self.queue) > self.__limit__:
            self.queue.pop(0)

        return True

    def length(self):
        return len(self.queue)

    def limit(self, s=None):
        if s is None:
            return self.__limit__

        self.__limit__ = s if s >= 0 else 0
