#coding=u8

import numpy as np
import common

def get_split_attr(X_set, Y_set, continuous_attrs):
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

class Decision_tree_C4_5():
    #内部节点类
    class Tree_node():
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

    __node_list = []
    
    def __init__(self, X_set, Y_set, continuous_attrs = []):
        self.__gen_tree_node(X_set, Y_set, continuous_attrs)

    def __gen_tree_node(self, X_set, Y_set, continuous_attrs):
        '''
        生成树节点
        '''
        tree_node = self.Tree_node()
        self.__node_list.append(tree_node)
        if common.is_same_element(Y_set):
            tree_node.node_type = 3
            tree_node.leaf_value = Y_set[0]
            return
        if common.is_same_dataframe(X_set):
            tree_node.node_type = 3
            tree_node.leaf_value = common.majorty_in_array(Y_set)
            return
        
        attr, split_value = get_split_attr(X_set, Y_set, continuous_attrs)
        tree_node.split_attr = attr
        #离散属性分割点,枚举分割属性直接将此属性从分割点中去除
        if split_value == None:
            tree_node.node_type = 1
            attr_values = X_set[attr].unique()
            child_node = {}
            for value in attr_values:
                child_node[value] = len(self.__node_list)
                index = np.array(X_set[attr] == value)
                new_X_set = X_set.iloc[index].drop(attr, axis = 1)
                new_Y_set = Y_set[index]
                self.__gen_tree_node(new_X_set, new_Y_set, continuous_attrs)
            tree_node.child_node = child_node
        #连续值属性分割点
        else:
            tree_node.node_type = 2
            tree_node.split_value = split_value
            #生成左树
            tree_node.left_node = len(self.__node_list)
            index = np.array(X_set[attr] <= split_value)
            new_X_set = X_set.iloc[index]
            new_Y_set = Y_set[index]
            self.__gen_tree_node(new_X_set, new_Y_set, continuous_attrs)
            #生成右树
            tree_node.right_node = len(self.__node_list)
            index = np.array(X_set[attr] > split_value)
            new_X_set = X_set.iloc[index]
            new_Y_set = Y_set[index]
            self.__gen_tree_node(new_X_set, new_Y_set, continuous_attrs)
            

    def dump_tree(self):
        self.__recursion_dump(self.__node_list[0], 0)

    def __recursion_dump(self, current_node, deep):
        #叶子节点
        if current_node.node_type == 3:
            print('\t' * deep, end = '')
            print('leaf node: {}'.format(current_node.leaf_value))
        #离散节点
        elif current_node.node_type == 1:
            for value, next_node in current_node.child_node.items():
                print('\t' * deep, end = '')
                print('{} = {} to {}'.format(current_node.split_attr, value, next_node))
                self.__recursion_dump(self.__node_list[next_node], deep + 1)
        elif current_node.node_type == 2:
            print('\t' * deep, end = '')
            print('{} <= {} to {}'.format(current_node.split_attr, current_node.split_value, current_node.left_node))
            self.__recursion_dump(self.__node_list[current_node.left_node], deep + 1)
            print('\t' * deep, end = '')
            print('{} > {} to {}'.format(current_node.split_attr, current_node.split_value, current_node.right_node))
            self.__recursion_dump(self.__node_list[current_node.right_node], deep + 1)
            
            
            
