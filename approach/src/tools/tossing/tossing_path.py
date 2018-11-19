class TossingPath:

    def __init__(self, bug_id, final_fixer, topic=0):
        # path contains tuples of developer name and timestamp?
        self._path = []
        self._bug_id = bug_id
        self._ffixer = final_fixer
        self._bug_topic = topic

    def add_path_item(self, path_item):
        self._path.append(path_item)

    def get_id_path(self):
        return [item[0] for item in self._path]

    def get_actual_path(self, dev_dict):
        ret = []
        for tp in self._path:
            ret.append(dev_dict[tp[0]])

        if len(ret) == 0:
            raise Exception('Tried to get length of an empty path.')
        return ret

    def get_goal_oriented_path(self, dev_dict):
        tmp = []
        if len(self._path) > 1:
            tmp.append(dev_dict[self._path[0][0]])
            tmp.append(dev_dict[self._path[-1][0]])
        return tmp
