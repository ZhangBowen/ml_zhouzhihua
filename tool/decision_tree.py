#coding=u8

import numpy as np
import common
import copy
'''
各种决策树实现
'''


class Decision_tree_C4_5():
    #内部节点类
    class Tree_node():
        #节点编号
        node_number = None
        '''
        节点类型:1 离散型节点;2 连续型节点; 3叶子节点
        '''
        node_type = None
        
        leaf_value = None

        #分割属性名
        split_attr = None

        #连续分割点
        split_value = None
        left_node = None
        right_node = None

        #离散分割点
        child_node = None

        #标签集合
        label_set = None

    _node_list = []

    def predict(self, X, tree = None):
        '''
        给出单个样本的预测结果
        '''
        if tree == None:
            tree = self._node_list

        if len(tree) == 0:
            return -1

        index = 0
        while tree[index].node_type != 3:
            current_node = tree[index]
            if current_node.node_type == 1:
                index = current_node.child_node.get(X[current_node.split_attr], list(current_node.child_node.values())[0])
            elif current_node.node_type == 2:
                value = X[current_node.split_attr]
                if value <= current_node.split_value:
                    index = current_node.left_node
                else:
                    index = current_node.right_node
            else:
                return -1

        return tree[index].leaf_value

    def evaluate(self, X_test, y_test, tree = None):
        '''
        使用树结构进行预测,给出准确率
        '''
        if tree == None:
            tree = self._node_list
        
        predict_reslut = list(map(self.predict, X_test.T.to_dict().values()))
        
        return len(y_test[y_test == predict_reslut]) * 1.0 / len(y_test)
    
    def __init__(self, X_set, Y_set, continuous_attrs = []):
        self._gen_tree_node(X_set, Y_set, continuous_attrs)

    def _get_split_attr(self, X_set, Y_set, continuous_attrs):
        '''
        依照信息熵增益选择分割节点
        '''
        old_entropy = common.get_entropy(Y_set)
        total_num = len(Y_set)
        max_gain = 0
        max_attr = None
        max_split_point = None
        for attr in X_set.columns:
            if attr in continuous_attrs:
                #连续值处理
                attr_value = np.sort(np.array(X_set[attr].unique()))
                for i in range(1, total_num):
                    split_point = (attr_value[i - 1] + attr_value[i]) / 2
                    #计算此分割点的信息增益
                    tmp_entropy = 0
                    
                    index = np.array(X_set[attr] <= split_point)
                    new_Y = Y_set[index]
                    new_num = len(new_Y)
                    tmp_entropy += common.get_entropy(new_Y) * float(new_num) / total_num

                    index = np.array(X_set[attr] > split_point)
                    new_Y = Y_set[index]
                    new_num = len(new_Y)
                    tmp_entropy += common.get_entropy(new_Y) * float(new_num) / total_num

                    info_gain = old_entropy - tmp_entropy

                    if info_gain > max_gain:
                        max_gain = info_gain
                        max_attr = attr
                        max_split_point = split_point
            else:
                #离散值处理
                new_entropy = 0
                done_attr = []
                for value in X_set.loc[:, attr]:
                    if value in done_attr:
                        continue
                    done_attr.append(value)
                    index = np.array(X_set[attr] == value)
                    new_Y = Y_set[index]
                    new_num = len(new_Y)
                    new_entropy += common.get_entropy(new_Y) * float(new_num) / total_num
                info_gain = old_entropy - new_entropy
                if info_gain > max_gain:
                    max_gain = info_gain
                    max_attr = attr
                    max_split_point = None
        
        return max_attr, max_split_point 

    def _make_leaf(self, Y_set):
        ''' hold住决策树的展开,暂将数据集攒成一个点 直接按照样本标签数量多寡决定标签的值 '''
        tree_node = self.Tree_node()
        tree_node.node_type = 3
        tree_node.leaf_value = common.majority_in_array(Y_set)
        
        return tree_node

    def _gen_tree_node(self, X_set, Y_set, continuous_attrs):
        '''
        生成树节点
        '''
        tree_node = self.Tree_node()
        tree_node.node_number = len(self._node_list)
        tree_node.label_set = Y_set
        self._node_list.append(tree_node)
        if common.is_same_element(Y_set):
            tree_node.node_type = 3
            tree_node.leaf_value = Y_set[0]
            return
        if common.is_same_dataframe(X_set):
            tree_node.node_type = 3
            tree_node.leaf_value = common.majority_in_array(Y_set)
            return
        
        attr, split_value = self._get_split_attr(X_set, Y_set, continuous_attrs)
        tree_node.split_attr = attr
        #离散属性分割点,枚举分割属性直接将此属性从分割点中去除
        if split_value == None:
            tree_node.node_type = 1
            attr_values = X_set[attr].unique()
            child_node = {}
            for value in attr_values:
                child_node[value] = len(self._node_list)
                index = np.array(X_set[attr] == value)
                new_X_set = X_set.iloc[index].drop(attr, axis = 1)
                new_Y_set = Y_set[index]
                self._gen_tree_node(new_X_set, new_Y_set, continuous_attrs)
            tree_node.child_node = child_node
        #连续值属性分割点
        else:
            tree_node.node_type = 2
            tree_node.split_value = split_value
            #生成左树
            tree_node.left_node = len(self._node_list)
            index = np.array(X_set[attr] <= split_value)
            new_X_set = X_set.iloc[index]
            new_Y_set = Y_set[index]
            self._gen_tree_node(new_X_set, new_Y_set, continuous_attrs)
            #生成右树
            tree_node.right_node = len(self._node_list)
            index = np.array(X_set[attr] > split_value)
            new_X_set = X_set.iloc[index]
            new_Y_set = Y_set[index]
            self._gen_tree_node(new_X_set, new_Y_set, continuous_attrs)
            

    def dump_tree(self):
        self._recursion_dump(self._node_list[0], 0)

    def _recursion_dump(self, current_node, deep):
        #叶子节点
        if current_node.node_type == 3:
            print('\t' * deep, end = '')
            print('{}, leaf node: {}'.format(current_node.node_number, current_node.leaf_value))
        #离散节点
        elif current_node.node_type == 1:
            for value, next_node in current_node.child_node.items():
                print('\t' * deep, end = '')
                print('{}, {} = {} to {}'.format(current_node.node_number, current_node.split_attr, value, next_node))
                self._recursion_dump(self._node_list[next_node], deep + 1)
        elif current_node.node_type == 2:
            print('\t' * deep, end = '')
            print('{}, {} <= {} to {}'.format(current_node.node_number, current_node.split_attr, current_node.split_value, current_node.left_node))
            self._recursion_dump(self._node_list[current_node.left_node], deep + 1)
            print('\t' * deep, end = '')
            print('{}, {} > {} to {}'.format(current_node.node_number, current_node.split_attr, current_node.split_value, current_node.right_node))
            self._recursion_dump(self._node_list[current_node.right_node], deep + 1)
            
class CART(Decision_tree_C4_5):
    def _get_split_attr(self, X_set, Y_set, continuous_attrs):
        '''
        依照信息熵增益选择分割节点
        '''
        old_entropy = common.get_Gini(Y_set)
        total_num = len(Y_set)
        max_gain = 0
        max_attr = None
        max_split_point = None
        for attr in X_set.columns:
            if attr in continuous_attrs:
                #连续值处理
                attr_value = np.sort(np.array(X_set[attr].unique()))
                for i in range(1, total_num):
                    split_point = (attr_value[i - 1] + attr_value[i]) / 2
                    #计算此分割点的信息增益
                    tmp_entropy = 0
                    
                    index = np.array(X_set[attr] <= split_point)
                    new_Y = Y_set[index]
                    new_num = len(new_Y)
                    tmp_entropy += common.get_Gini(new_Y) * float(new_num) / total_num

                    index = np.array(X_set[attr] > split_point)
                    new_Y = Y_set[index]
                    new_num = len(new_Y)
                    tmp_entropy += common.get_Gini(new_Y) * float(new_num) / total_num

                    info_gain = old_entropy - tmp_entropy

                    if info_gain > max_gain:
                        max_gain = info_gain
                        max_attr = attr
                        max_split_point = split_point
            else:
                #离散值处理
                new_entropy = 0
                done_attr = []
                for value in X_set.loc[:, attr]:
                    if value in done_attr:
                        continue
                    done_attr.append(value)
                    index = np.array(X_set[attr] == value)
                    new_Y = Y_set[index]
                    new_num = len(new_Y)
                    new_entropy += common.get_Gini(new_Y) * float(new_num) / total_num
                info_gain = old_entropy - new_entropy
                if info_gain > max_gain:
                    max_gain = info_gain
                    max_attr = attr
                    max_split_point = None
        
        return max_attr, max_split_point 

class Pre_pruning_DT(Decision_tree_C4_5):
    '''
    预剪枝决策树
    '''

    #剪枝策略集合
    X_test = None
    y_test = None

    _node_list = []

    def __init__(self, X_train, y_train, X_test, y_test, continuous_attrs = []):
        self.X_test = X_test
        self.y_test = y_test
        
        #展开树
        tree_node = self._make_leaf(y_train)
        tree_node.node_number = 0
        self._node_list.append(tree_node)
        self._gen_tree_node(X_train, y_train, continuous_attrs, self._node_list[0])

    def _gen_tree_node(self, X_set, Y_set, continuous_attrs, tree_node):
        '''
        展开树节点
        '''
        if common.is_same_element(Y_set):
            return
        if common.is_same_dataframe(X_set):
            return

        #剪枝,首先计算展开节点之前的准确率
        old_accuracy_ratio = self.evaluate(self.X_test, self.y_test)
        #对节点列表做备份
        node_list_backup = copy.deepcopy(self._node_list)

        #分裂属性选择
        attr, split_value = self._get_split_attr(X_set, Y_set, continuous_attrs)
        tree_node.split_attr = attr

        #待分裂节点下标
        split_index_begin = len(self._node_list)
        split_index_end = split_index_begin

        #离散属性分割点,枚举分割属性直接将此属性从分割点中去除
        if split_value == None:
            tree_node.node_type = 1
            attr_values = X_set[attr].unique()
            child_node = {}
            for value in attr_values:
                child_node[value] = split_index_end
                index = np.array(X_set[attr] == value)
                new_Y_set = Y_set[index]
                new_node = self._make_leaf(new_Y_set)
                new_node.node_number = split_index_end
                self._node_list.append(new_node)
                split_index_end += 1
            tree_node.child_node = child_node
        #连续值属性分割点
        else:
            tree_node.node_type = 2
            tree_node.split_value = split_value
            #生成左树
            tree_node.left_node = split_index_end
            index = np.array(X_set[attr] <= split_value)
            new_Y_set = Y_set[index]
            new_node = self._make_leaf(new_Y_set)
            new_node.node_number = split_index_end
            self._node_list.append(new_node)
            split_index_end += 1
            
            #生成右树
            tree_node.right_node = split_index_end
            index = np.array(X_set[attr] > split_value)
            new_Y_set = Y_set[index]
            new_node = self._make_leaf(new_Y_set)
            new_node.node_number = split_index_end
            self._node_list.append(new_node)
            split_index_end += 1

        #比较展开前后错误率,剪枝
        new_accuracy_ratio = self.evaluate(self.X_test, self.y_test)
        if new_accuracy_ratio < old_accuracy_ratio:
            print("new accuracy {:.2f} is lower than old accuracy {:.2f}. withdraw segmentation".format(
                new_accuracy_ratio, old_accuracy_ratio))
            self._node_list = node_list_backup
            return
        

        #效果提升,展开后续节点
        if split_value == None:
            attr_values = X_set[attr].unique()
            for value in attr_values:
                index = np.array(X_set[attr] == value)
                new_X_set = X_set.iloc[index].drop(attr, axis = 1)
                new_Y_set = Y_set[index]
                self._gen_tree_node(new_X_set, new_Y_set, continuous_attrs, self._node_list[split_index_begin])
                split_index_begin += 1
        #连续值属性分割点
        else:
            #生成左树
            index = np.array(X_set[attr] <= split_value)
            new_X_set = X_set.iloc[index]
            new_Y_set = Y_set[index]
            self._gen_tree_node(new_X_set, new_Y_set, continuous_attrs, self._node_list[split_index_begin])
            #生成右树
            index = np.array(X_set[attr] > split_value)
            new_X_set = X_set.iloc[index]
            new_Y_set = Y_set[index]
            self._gen_tree_node(new_X_set, new_Y_set, continuous_attrs, self._node_list[split_index_begin + 1])

class Post_pruning_DT(Decision_tree_C4_5):
    '''
    后剪枝决策树
    '''

    #剪枝策略集合
    X_test = None
    y_test = None

    _node_list = []

    def __init__(self, X_train, y_train, X_test, y_test, continuous_attrs = []):
        self.X_test = X_test
        self.y_test = y_test
        
        self._gen_tree_node(X_train, y_train, continuous_attrs)
        #后剪枝
        self._post_pruning()

    def _post_pruning(self):
        old_accuracy = self.evaluate(self.X_test, self.y_test)
        #倒序遍历
        for i in range(len(self._node_list) - 1, -1, -1):
            cn = self._node_list[i]
            if cn.node_type == 3:
                continue
            back_up = copy.deepcopy(cn)
            #合并为叶子节点
            cn.node_type = 3
            cn.leaf_value = common.majority_in_array(cn.label_set)

            new_accuracy = self.evaluate(self.X_test, self.y_test)
            if new_accuracy > old_accuracy:
                print("new accuracy {:.2f} is higher than old accuracy {:.2f}. merge node".format(
                    new_accuracy, old_accuracy))
                old_accuracy = new_accuracy
            else:
                #还原
                cn.node_type = back_up.node_type
