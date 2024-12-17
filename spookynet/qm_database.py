import os
import apsw
import numpy as np
import multiprocessing
from torch.utils.data import Dataset
from ase import Atoms
from ase.neighborlist import neighbor_list
from sklearn.neighbors import BallTree
import numpy as np
import torch

class QMDatabase(Dataset):
    def __init__(self, filename, flags=apsw.SQLITE_OPEN_READONLY):
        super().__init__()
        self.db = filename
        self.connections = {} #allow multiple connections (needed for multi-threading)
        self._open(flags=flags) #creates the database if it doesn't exist yet

    def __len__(self):
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        return cursor.execute('''SELECT * FROM metadata WHERE id=1''').fetchone()[-1]

    def __getitem__(self, idx):
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        try:
            if isinstance(idx, list):  # for batched data retrieval
                if not idx:  # check if the list is empty
                    return []
                # Use parameterized query to prevent SQL injection
                placeholders = ','.join('?' for _ in idx)
                query = f'SELECT * FROM data WHERE id IN ({placeholders})'
                data = cursor.execute(query, idx).fetchall()
                return [self._unpack_data_tuple(i) for i in data]
            else:
                # Use parameterized query to prevent SQL injection and ensure correct data type
                query = 'SELECT * FROM data WHERE id=?'
                data = cursor.execute(query, (idx,)).fetchone()
                if data is None:
                    raise IndexError(f"No data found for index {idx}")
                return self._unpack_data_tuple(data)
        except Exception as e:
            raise RuntimeError(f"Database error: {e}")

    def _unpack_data_tuple(self, data):
        def get_idx(atomic_numbers, coordinates, cutoff=6, pbc=False, cell=None):
            atoms = Atoms(numbers=atomic_numbers, positions=coordinates, pbc=pbc, cell=cell)
            N = len(atoms)
            positions = np.copy(atoms.positions)
            if pbc:
                idx_i, idx_j, cell_offsets = neighbor_list("ijS", atoms, cutoff)
                idx_i = torch.tensor(idx_i, dtype=torch.int32)
                idx_j = torch.tensor(idx_j, dtype=torch.int32)
                cell_offsets = torch.tensor(cell_offsets, dtype=torch.float32)
            else:
                tree = BallTree(positions)
                idx_i = []
                idx_j = tree.query_radius(positions, r=cutoff)
                for i in range(len(idx_j)):
                    idx = idx_j[i]  # all neighbors with self-interaction
                    idx = idx[idx != i]  # filter out self-interaction
                    idx_i.append(np.full(idx.shape, i, idx.dtype))
                    idx_j[i] = idx
                idx_i = torch.tensor(np.concatenate(idx_i), dtype=torch.int32)
                idx_j = torch.tensor(np.concatenate(idx_j), dtype=torch.int32)
                cell_offsets = False
            return cell, cell_offsets, idx_i, idx_j
        # print(data[5])
        N = len(data[3]) // 4  # a single int32 is 4 bytes
        Q = torch.tensor([0.0 if data[1] is None else data[1]], dtype=torch.float32)
        S = torch.tensor([0.0 if data[2] is None else data[2]], dtype=torch.float32)
        Z = torch.tensor(self._deblob(data[3], dtype=np.int32, shape=(N,)), dtype=torch.int32)
        R = torch.tensor(self._deblob(data[4], dtype=np.float32, shape=(N, 3)), dtype=torch.float32)
        E = torch.tensor([0.0 if data[5] is None else data[5]], dtype=torch.float32)
        F = torch.tensor(self._deblob(data[6], dtype=np.float32, shape=(N, 3)), dtype=torch.float32)
        D = torch.tensor(self._deblob(data[7], dtype=np.float32, shape=(1, 3)), dtype=torch.float32)
        cell = torch.tensor(self._deblob(data[8], dtype=np.float32, shape=(3, 3)), dtype=torch.float32) if len(data) > 8 else [False, False, False]

        pbc = True if any(cell) else False
        cell, cell_offsets, idx_i, idx_j = get_idx(Z, R, 6, pbc, cell)
        return Q, S, Z, R, E, F, D, idx_i, idx_j, cell, cell_offsets

    def add_data(self, Q, S, Z, R, E, F, D, cell, flags=apsw.SQLITE_OPEN_READWRITE, transaction=True):
        #check that no NaN values are added
        if self._any_is_nan(Q, S, Z, R, E, F, D, cell):
            print("encountered NaN, data is not added")
            return
        cursor = self._get_connection(flags=flags).cursor()
        #update data
        if transaction:
            #begin exclusive transaction (locks db) which is necessary
            #if database is accessed from multiple programs at once (default for safety)
            cursor.execute('''BEGIN EXCLUSIVE''')
        try:
            length = cursor.execute('''SELECT * FROM metadata WHERE id=1''').fetchone()[-1]
            cursor.execute('''INSERT INTO data (id, Q, S, Z, R, E, F, D, cell) VALUES (?,?,?,?,?,?,?,?,?)''',
                (None if length > 0 else 0, #autoincrementing ID
                    None if Q is None else float(Q), None if S is None else float(S),
                    self._blob(Z), self._blob(R),
                    None if E is None else float(E), self._blob(F), self._blob(D), self._blob(cell)))
            #update metadata
            cursor.execute('''INSERT OR REPLACE INTO metadata VALUES (?,?)''', (1, length+1))
            Nmax = cursor.execute('''SELECT * FROM metadata WHERE id=0''').fetchone()[-1]
            if Z.shape[0] > Nmax: #if N > Nmax
                cursor.execute('''INSERT OR REPLACE INTO metadata VALUES (?,?)''', (0, Z.shape[0]))
            if transaction:
                cursor.execute('''COMMIT''') #end transaction
        except Exception as exc:
            if transaction:
                cursor.execute('''ROLLBACK''')
            raise exc

    def _any_is_nan(self, *vals):
        nan = False
        for val in vals:
            if val is None:
                continue
            nan = nan or np.any(np.isnan(val))
        return nan

    def _blob(self, array):
        """Convert numpy array to blob/buffer object."""
        if array is None:
            return None
        if array.dtype == np.float64:
            array = array.astype(np.float32)
        if array.dtype == np.int64:
            array = array.astype(np.int32)
        if not np.little_endian:
            array = array.byteswap()
        return memoryview(np.ascontiguousarray(array))

    def _deblob(self, buf, dtype=np.float32, shape=None):
        """Convert blob/buffer object to numpy array."""
        if buf is None:
            return np.zeros(shape)
        array = np.frombuffer(buf, dtype)
        if not np.little_endian:
            array = array.byteswap()
        array.shape = shape
        return np.copy(array)

    def _open(self, flags=apsw.SQLITE_OPEN_READONLY):
        newdb = not os.path.isfile(self.db)
        cursor = self._get_connection(flags=flags).cursor()
        if newdb:
            #create table to store data
            cursor.execute('''CREATE TABLE IF NOT EXISTS data
                (id INTEGER NOT NULL PRIMARY KEY,
                 Q FLOAT,
                 S FLOAT,
                 Z BLOB,
                 R BLOB,
                 E FLOAT,
                 F BLOB,
                 D BLOB,
                 cell BLOB
                )''')
            #create table to store metadata (information about Nmax and the length, i.e. number of entries)
            cursor.execute('''CREATE TABLE IF NOT EXISTS metadata
                (id INTEGER PRIMARY KEY, N INTEGER)''')
            #meta data values are created
            cursor.execute('''INSERT OR IGNORE INTO metadata (id, N) VALUES (?,?)''', (0, 0)) #Nmax
            cursor.execute('''INSERT OR IGNORE INTO metadata (id, N) VALUES (?,?)''', (1, 0)) #num_data

    def _get_connection(self, flags=apsw.SQLITE_OPEN_READONLY):
        '''
        This allows multiple processes to access the database at once,
        every process must have its own connection
        '''
        key = multiprocessing.current_process().name
        if key not in self.connections.keys():
            self.connections[key] = apsw.Connection(self.db, flags=flags)
            self.connections[key].setbusytimeout(300000) #5 minute timeout
        return self.connections[key]

    @property
    def Nmax(self):
        cursor = self._get_connection(flags=apsw.SQLITE_OPEN_READONLY).cursor()
        return cursor.execute('''SELECT * FROM metadata WHERE id=0''').fetchone()[-1]

    def write_xyz(self, idx, filename=None):
        if filename == None:
            filename = str(idx)+".xyz"
        Q, S, Z, R, E, F, D, cell = self[idx]
        with open(filename, "w") as file:
            file.write(str(Z.shape[0])+"\n")
            file.write("Q: {0} S: {1} E: {2: 15.6f} D: {3: 11.6f} {4: 11.6f} {5: 11.6f}\n".format(int(Q[0]), int(S[0]), E[0], D[0,0], D[0,1], D[0,2]))
            for z, r, f in zip(Z, R, F):
                file.write('{0} {1: 11.6f} {2: 11.6f} {3: 11.6f} {4: 11.6f} {5: 11.6f} {6: 11.6f}\n'.format(_labels[z], *r, *f))

#this is for generating xyz files
_labels = ['n ', 'H ', 'He',
    'Li', 'Be', 'B ', 'C ', 'N ', 'O ', 'F ', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P ', 'S ',
    'Cl', 'Ar', 'K ', 'Ca', 'Sc', 'Ti', 'V ', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y ', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
    'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I ', 'Xe', 'Cs', 'Ba', 'La', 'Ce',
    'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf',
    'Ta', 'W ', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn']
