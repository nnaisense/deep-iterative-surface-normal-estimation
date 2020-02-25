import torch
import scipy.spatial


def radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32):
    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    assert x.dim() == 2 and batch_x.dim() == 1
    assert y.dim() == 2 and batch_y.dim() == 1
    assert x.size(1) == y.size(1)
    assert x.size(0) == batch_x.size(0)
    assert y.size(0) == batch_y.size(0)

    rows = []
    cols = []
    denses = []
    current_first = 0

    for element in torch.unique(batch_x, sorted=True):
        x_element = x[batch_x == element]
        y_element = y[batch_y == element]
        tree = scipy.spatial.cKDTree(x_element)
        _, col = tree.query(y_element, k=max_num_neighbors, distance_upper_bound=r + 1e-8, eps=1e-8)
        col = [torch.tensor(c) for c in col]
        row = [torch.full_like(c, i) for i, c in enumerate(col)]
        row, col = torch.cat(row, dim=0), torch.cat(col, dim=0)

        # Hack: Fill missing neighbors with self loops, if there are any
        # missing = (col == tree.n).nonzero()
        col[col == tree.n] = row[col == tree.n]
        dense = col.view(-1, max_num_neighbors)
        col = col + current_first
        row = row + current_first
        dense = dense + current_first
        current_first += x_element.size(0)
        rows.append(row)
        cols.append(col)
        denses.append(dense)

    row = torch.cat(rows, dim=0)
    col = torch.cat(cols, dim=0)
    dense = torch.cat(denses, dim=0)
    return torch.stack([row, col], dim=0), dense  # , missing


def radius_graph(x, r, batch=None, max_num_neighbors=32):
    return radius(x, x, r, batch, batch, max_num_neighbors + 1)


def radius_var(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32):
    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    assert x.dim() == 2 and batch_x.dim() == 1
    assert y.dim() == 2 and batch_y.dim() == 1
    assert x.size(1) == y.size(1)
    assert x.size(0) == batch_x.size(0)
    assert y.size(0) == batch_y.size(0)

    rows = []
    cols = []
    denses = []
    current_first = 0

    for element in torch.unique(batch_x, sorted=True):
        x_element = x[batch_x == element]
        y_element = y[batch_y == element]
        tree = scipy.spatial.cKDTree(x_element)
        _, col = tree.query(y_element, k=max_num_neighbors, distance_upper_bound=r + 1e-8, eps=1e-8)
        col = [torch.tensor(c) for c in col]
        col = [c[c != tree.n] for c in col]
        row = [torch.full_like(c, i) for i, c in enumerate(col)]
        row, col = torch.cat(row, dim=0), torch.cat(col, dim=0)

        # Hack: Fill missing neighbors with self loops, if there are any
        # missing = (col == tree.n).nonzero()
        col = col + current_first
        row = row + current_first
        current_first += x_element.size(0)
        rows.append(row)
        cols.append(col)

    row = torch.cat(rows, dim=0)
    col = torch.cat(cols, dim=0)
    return torch.stack([row, col], dim=0)  # , missing


def radius_graph_var(x, r, batch=None, max_num_neighbors=32):
    return radius_var(x, x, r, batch, batch, max_num_neighbors + 1)
