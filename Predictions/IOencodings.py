import torch


class FixedSizeEncoding():
    '''
    Encodes inputs and outputs using a fixed size for each input

    For now we only encode lists of stuff

    nb_arguments_max: maximum number of inputs
    size_max: maximum number of elements in an input (= list)
    lexicon: list of symbols that can appear (for instance range(-10,10))

    Example:
    IO = [[I1, I2, ..., Ik], O] 
    size_max = 2
    nb_arguments_max = 3 
    IO = [[[11,20],[3]], [12,2]] 
    ie I1 = [11,20], I2 = [3], O = [12,2]
    the encoding is (ignoring the symbolToIndex)
    [11,1,20,1,3,1,0,0,0,0,0,0, 12,1,2,1]
    every second position is 1 or 0, with 0 meaning "padding" 
    '''

    def __init__(self,
                 nb_arguments_max,
                 lexicon,
                 size_max,
                 ) -> None:
        self.nb_arguments_max = nb_arguments_max
        self.size_max = size_max
        self.output_dimension = 2 * size_max * (1 + nb_arguments_max)
        self.lexicon = lexicon[:]  # Make a copy since we modify it in place
        self.lexicon += ["PAD", "NOTPAD"]
        self.lexicon_size = len(self.lexicon)
        self.symbolToIndex = {
            symbol: index for index, symbol in enumerate(self.lexicon)
        }

    def _encode_single_arg(self, arg):
        '''
        encodes a single list (representing an input or an output)
        '''
        if isinstance(arg, int):
            arg = [arg]
        res = torch.zeros(2*self.size_max, dtype=torch.long)
        res += self.symbolToIndex["PAD"]
        if len(arg) > self.size_max:
            assert False, \
                "IOEncodings.py: FixedSizeEncoding: This input is too long: len({})={} > {}".format(arg, len(arg), self.size_max)
        for i, e in enumerate(arg):
            res[2*i] = self.symbolToIndex[e]
            # Boolean flag: the previous value is not padding
            res[2*i+1] = self.symbolToIndex["PAD"]
        return res

    def encode_IO(self, IO):
        '''
        embeds a list of inputs and its associated output
        IO is of the form [[I1,I2, ..., Ik], O] 
        where I1, I2, ..., Ik are inputs and O an output 

        outputs a tensor of dimension self.output_dimension
        '''
        res = []
        inputs, output = IO
        if len(inputs) > self.nb_arguments_max:
            assert False, \
                "IOEncodings.py: FixedSizeEncoding: Too many inputs: len({})={} > {}".format(
                    inputs, len(inputs), self.nb_arguments_max)
        for i in range(self.nb_arguments_max):
            try:
                input_ = inputs[i]
                embedded_input = self._encode_single_arg(input_)
                res.append(embedded_input)
            except:
                not_pad_tensor = torch.zeros(2*self.size_max, dtype=torch.long)
                not_pad_tensor += self.symbolToIndex["PAD"]
                res.append(not_pad_tensor)
        res.append(self._encode_single_arg(output))
        res = torch.cat(res)
        # assert(len(res) == self.output_dimension)
        return res

    def encode_IOs(self, IOs):
        '''
        encodes a list of IOs by stacking

        outputs a tensor of dimension 
        len(IOs) * self.output_dimension
        '''
        res = []
        for IO in IOs:
            res.append(self.encode_IO(IO))
        res = torch.stack(res)
        return res


class VariableSizeEncoding():
    '''
    Encodes inputs and outputs using a variable size and separators

    For now we only encode List(INT)

    nb_arguments_max: maximum number of inputs
    lexicon: all elements of a list must be from lexicon
    output_dimension: output dimension of the encoding

    Example:
    IO = [[I1, I1, ..., Ik], O] 
    output_dimension = 15
    nb_arguments_max = 3 
    IO = [[[11,20], [3]], [12,2]] 
    ie I1 = [11,20], I2 = [3], O = [12,2]
    the encoding is (modulo SymbolToIndex encoding of the symbols)
    tensor(["STARTING", 11,20, ENDOFINPUT", 3, "ENDOFINPUT", "STARTOFOUTPUT", 12,2,
    "ENDING", "ENDING", "ENDING", "ENDING", "ENDING", "ENDING"])
    '''

    def __init__(self,
                 nb_arguments_max,
                 lexicon,
                 output_dimension,
                 ) -> None:
        self.nb_arguments_max = nb_arguments_max
        self.output_dimension = output_dimension

        self.specialSymbols = [
            "STARTING",  # start of entire sequence
            "ENDOFINPUT",  # delimits the ending of an input - we might have multiple inputs
            "STARTOFOUTPUT",  # begins the start of the output
            "ENDING",  # ending of entire sequence
        ]
        self.lexicon = lexicon + self.specialSymbols
        self.lexicon_size = len(lexicon)
        self.symbolToIndex = {
            symbol: index for index, symbol in enumerate(self.lexicon)
        }
        self.startingIndex = self.symbolToIndex["STARTING"]
        self.endOfInputIndex = self.symbolToIndex["ENDOFINPUT"]
        self.startOfOutputIndex = self.symbolToIndex["STARTOFOUTPUT"]
        self.endingIndex = self.symbolToIndex["ENDING"]

    def encode_IO(self, IO):
        '''
        embed a list of inputs and its associated output
        IO is of the form [[I1, I2, ..., Ik], O] 
        where I1, I2, ..., Ik are inputs and O is an output

        outputs a tensor of dimension self.output_dimension
        '''
        e = [self.startingIndex]
        inputs, output = IO
        size = 0
        for x in inputs:
            for s in x:
                e.append(self.symbolToIndex[s])
                size += 1
            e.append(self.endOfInputIndex)
            size += 1
        e.append(self.startOfOutputIndex)
        size += 1
        for s in output:
            e.append(self.symbolToIndex[s])
            size += 1
        if size > self.output_dimension - 2:
            assert False, \
                "IOEncodings.py: VariableSizeEncoding: IO too large: {} > {}-2 for {}".format(size, self.output_dimension, IO)
        else:
            for _ in range(self.output_dimension - size - 1):
                e.append(self.endingIndex)
        res = torch.LongTensor(e)
        # assert(len(res) == self.output_dimension)
        return res

    def encode_IOs(self, IOs):
        '''
        encodes a list of IOs by stacking

        outputs a tensor of dimension 
        len(IOs) * self.output_dimension
        '''
        IOs = sorted(IOs, key=lambda xs_y: sum(
            len(z) + 1 for z in xs_y[0]) + len(xs_y[1]), reverse=True)

        res = [None] * len(IOs)
        for j, IO in enumerate(IOs):
            res[j] = self.encode_IO(IO)

        res = torch.stack(res)
        return res

class ZendoStructureEncoding:
    def __init__(self, lexicon, max_objects):
        """
        :param lexicon: list of valid color/shape/orientation strings
        :param max_objects: max number of pieces in one scene
        """
        self.max_objects = max_objects

        self.directions = ["left", "right", "front", "back", "top", "bottom"]

        # Automatically expand the lexicon
        full_lexicon = lexicon + [f"ID_{i}" for i in range(max_objects)] + ["PAD", "NONE"]
        self.symbolToIndex = {symbol: idx for idx, symbol in enumerate(full_lexicon)}
        self.indexToSymbol = {idx: symbol for symbol, idx in self.symbolToIndex.items()}
        self.lexicon_size = len(self.symbolToIndex)
        self.lexicon = full_lexicon

        # Set padding and fallback tokens
        self.token_PAD = self.symbolToIndex["PAD"]
        self.token_NONE = self.symbolToIndex["NONE"]

        self.vector_length = (
            1 +  # ID (as int)
            1 +  # color
            1 +  # shape
            1 +  # orientation
            6 +  # touching (6 directions)
            1    # pointing
        )
        self.output_dimension = self.vector_length * self.max_objects


    def encode_piece(self, piece, id_map):
        vec = []

        # Encode piece ID
        idx = id_map.get(piece["ID"], -1)
        symbolic_id = f"ID_{idx}" if idx >= 0 else "NONE"
        vec.append(self.symbolToIndex.get(symbolic_id, self.token_NONE))

        # Encode color, shape, orientation
        for key in ["color", "shape", "orientation"]:
            val = piece.get(key, "NONE")
            vec.append(self.symbolToIndex.get(val, self.token_NONE))

        # Encode touching (6 directions)
        for dir in self.directions:
            target_id = piece.get("touching", {}).get(dir, None)
            if target_id and target_id in id_map:
                symbolic_target = f"ID_{id_map[target_id]}"
            else:
                symbolic_target = "NONE"
            vec.append(self.symbolToIndex.get(symbolic_target, self.token_NONE))

        # Encode pointing
        pointed_id = piece.get("pointing", "")
        if pointed_id and pointed_id in id_map:
            symbolic_pointed = f"ID_{id_map[pointed_id]}"
        else:
            symbolic_pointed = "NONE"
        vec.append(self.symbolToIndex.get(symbolic_pointed, self.token_NONE))

        # Final validation
        for i, v in enumerate(vec):
            if not isinstance(v, int) or v >= self.lexicon_size:
                raise ValueError(f"🚨 Invalid token index {v} at position {i} in {vec}")
        return torch.tensor(vec, dtype=torch.long)

    def encode_structure(self, structure):
        """
        Encodes a full Zendo scene (list of pieces) into a tensor.
        Pads to self.max_objects.
        """
        objects = structure["objects"]
        id_map = {
            obj['ID']: idx for idx, obj in enumerate(objects)
        }


        result = []

        for i in range(self.max_objects):
            if i < len(objects):
                vec = self.encode_piece(objects[i], id_map)
            else:
                vec = torch.full((self.vector_length,), self.token_PAD, dtype=torch.long)
            result.append(vec)

        return torch.cat(result)

    def encode_IO(self, IO):
        """
        IO = [input_structure_dict, label]
        Output is a flat tensor
        """
        structure, label = IO
        encoded_input = self.encode_structure(structure)
        encoded_output = torch.tensor([label], dtype=torch.long)
        return torch.cat([encoded_input, encoded_output])

    def encode_IOs(self, IOs):
        encoded = [self.encode_IO(io) for io in IOs]
        tensor = torch.stack(encoded)
        return tensor
