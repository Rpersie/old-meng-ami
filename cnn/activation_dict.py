# Dictionary to maintain a list of feature matrices (w/ associated names),
# sorted by the average magnitude of their activation for a given layer.
# Stores for multiple layers


class ActivationDictNode(object):
    def __init__(self, avg_activation, name, feats):
        super(ActivationDictNode, self).__init__()
        
        # Store initial parameters
        self.avg_activation = avg_activation
        self.name = name
        self.feats = feats

        # Start out as childless orphan
        # Christ, that's dark...
        self.parent = None
        self.left_child = None
        self.right_child = None
        
        # Track number of elements in subtrees
        self.left_count = 0
        self.right_count = 0

    def insert_child(self, node):
        if node.avg_activation >= self.avg_activation:
            if self.right_child is None:
                # Leaf node -- easy
                self.right_child = node
                node.parent = self
            else:
                # Internal node
                self.right_child.insert_child(node)
            self.right_count += 1
        else:
            if self.left_child is None:
                # Leaf node -- easy
                self.left_child = node
                node.parent = self
            else:
                # Internal node
                self.left_child.insert_child(node)
            self.left_count += 1

    # In-order reversal traversal
    # Return (name, feats) tuples from best to worst activation
    def ordered_entries(self):
        entries = []

        if self.right_child is not None:
            right_entries = self.right_child.ordered_entries()
            entries.extend(right_entries)
        entries.append((self.name, self.feats))
        if self.left_child is not None:
            left_entries = self.left_child.ordered_entries()
            entries.extend(left_entries)

        return entries

class ActivationDict(object):
    def __init__(self, layer_names=[], top_count=100):
        super(ActivationDict, self).__init__()

        # Store initial parameters
        self.layer_names = layer_names
        self.top_count = top_count
        
        # Set up binary search trees
        self.bst_heads = {layer_name: None for layer_name in self.layer_names}
        self.worst_activations = {layer_name: float('-inf') for layer_name in self.layer_names}

    def top_ordered(self, layer_name):
        # In-order traversal of BST
        bst_head = self.bst_heads[layer_name]
        if bst_head is None:
            return []
        else:
            return bst_head.ordered_entries()

    def update_top(self, layer_name, avg_activation, name, feats):
        # Only add if in top if we're full
        bst_head = self.bst_heads[layer_name]
        full = (bst_head is not None) and (bst_head.left_count + bst_head.right_count + 1 == self.top_count)
        if not full or avg_activation > self.worst_activations[layer_name]:
            new_node = ActivationDictNode(avg_activation, name, feats)

            if bst_head is not None:
                if full:
                    # Remove worst node
                    if bst_head.left_child is None:
                        # Unlikely, but possible
                        new_bst_head = bst_head.right_child
                        del bst_head
                        self.bst_heads[layer_name] = new_bst_head
                        bst_head = new_bst_head
                    else:
                        # Traverse left until leaf
                        parent = bst_head
                        while (parent.left_child.left_child is not None):
                            parent = parent.left_child
                        worst = parent.left_child
                        parent.left_child = worst.right_child
                        parent.left_count -= 1
                        if worst.right_child is not None:
                            worst.right_child.parent = parent
                        del worst 

                # Insert our new node
                bst_head.insert_child(new_node)
            else:
                # First element
                self.bst_heads[layer_name] = new_node

            # Find our new worst
            worst_node = self.bst_heads[layer_name]
            while worst_node.left_child is not None:
                worst_node = worst_node.left_child
            self.worst_activations[layer_name] = worst_node.avg_activation

def test_activation_dict():
    # First, instantiate a small example
    test_ad = ActivationDict(layer_names=["A", "B", "C"], top_count=3)
    test_idx = 1
    
    # Test 1: Empty activation dict
    expected = []
    actual = test_ad.top_ordered("A")
    if expected == actual:
        print("Test %d passed" % test_idx)
    else:
        print("Test %d FAILED: expected %s, got %s" % ((test_idx, str(expected), str(actual))))
    test_idx += 1
    
    # Test 2: One entry activation dict
    test_ad.update_top("B", 1.0, "Test 1", [1])
    expected = [("Test 1", [1])]
    actual = test_ad.top_ordered("B")
    if expected == actual:
        print("Test %d passed" % test_idx)
    else:
        print("Test %d FAILED: expected %s, got %s" % ((test_idx, str(expected), str(actual))))
    test_idx += 1
    
    # Test 3: Full entry activation dict
    test_ad.update_top("B", 3.0, "Test 2", [2])
    test_ad.update_top("B", 0.5, "Test 3", [3])
    expected = [("Test 2", [2]), ("Test 1", [1]), ("Test 3", [3])]
    actual = test_ad.top_ordered("B")
    if expected == actual:
        print("Test %d passed" % test_idx)
    else:
        print("Test %d FAILED: expected %s, got %s" % ((test_idx, str(expected), str(actual))))
    test_idx += 1
    
    # Test 4: Add element not in top
    test_ad.update_top("B", 0.1, "Test 4", [4])
    expected = [("Test 2", [2]), ("Test 1", [1]), ("Test 3", [3])]
    actual = test_ad.top_ordered("B")
    if expected == actual:
        print("Test %d passed" % test_idx)
    else:
        print("Test %d FAILED: expected %s, got %s" % ((test_idx, str(expected), str(actual))))
    test_idx += 1
    
    # Test 5: Add new least element
    test_ad.update_top("B", 0.6, "Test 5", [5])
    expected = [("Test 2", [2]), ("Test 1", [1]), ("Test 5", [5])]
    actual = test_ad.top_ordered("B")
    if expected == actual:
        print("Test %d passed" % test_idx)
    else:
        print("Test %d FAILED: expected %s, got %s" % ((test_idx, str(expected), str(actual))))
    test_idx += 1
    
    # Test 6: Add new internal element
    test_ad.update_top("B", 1.3, "Test 6", [6])
    expected = [("Test 2", [2]), ("Test 6", [6]), ("Test 1", [1])]
    actual = test_ad.top_ordered("B")
    if expected == actual:
        print("Test %d passed" % test_idx)
    else:
        print("Test %d FAILED: expected %s, got %s" % ((test_idx, str(expected), str(actual))))
    test_idx += 1
    
    # Test 7: Add new best element
    test_ad.update_top("B", 10.0, "Test 7", [7])
    expected = [("Test 7", [7]), ("Test 2", [2]), ("Test 6", [6])]
    actual = test_ad.top_ordered("B")
    if expected == actual:
        print("Test %d passed" % test_idx)
    else:
        print("Test %d FAILED: expected %s, got %s" % ((test_idx, str(expected), str(actual))))
    test_idx += 1
