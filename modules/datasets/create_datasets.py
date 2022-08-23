from pathlib import Path
import numpy as np
from .utils import draw_text, get_position, get_fill, compose, save

# create A-B dataset
def create_AB(savepath = r"./data/datasets/AB", image_size=224, texture_path = r"./data/textures/train", seed=0):
    Path(savepath).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(200):
        # create A
        label = "A"
        image1, mask1 = draw_text("A", image_size, position=get_position(image_size, "", item_size=40, rng=rng), fontname="Roboto-Bold", fontsize=40, filling="KTH_cork", base_path=texture_path, rng=rng)
        image2, mask2 = draw_text("", image_size, position=get_position(image_size, "", item_size=40, rng=rng), fontname="Roboto-Bold", fontsize=40, filling="KTH_cork", base_path=texture_path, rng=rng)
        image3, mask3 = draw_text("+", image_size, position=get_position(image_size, "", item_size=80, rng=rng), fontname="Roboto-Bold", fontsize=80, filling="KTH_cotton", base_path=texture_path, rng=rng)
        image4, mask4 = get_fill("KTH_orange_peel", image_size, base_path=texture_path, rng=rng)
        image, masks = compose((image1, mask1),(image2, mask2),(image3, mask3),(image4, mask4))
        save(image, f"img_{label}_{i}", label, masks, basefolder = savepath, ground_truth = mask1)
        # create B
        label = "B"
        image1, mask1 = draw_text("", image_size, position=get_position(image_size, "", item_size=40, rng=rng), fontname="Roboto-Bold", fontsize=40, filling="G", base_path="", rng=rng)
        image2, mask2 = draw_text("B", image_size, position=get_position(image_size, "", item_size=40, rng=rng), fontname="Roboto-Bold", fontsize=40, filling="G", base_path="", rng=rng)
        image3, mask3 = draw_text("+", image_size, position=get_position(image_size, "", item_size=80, rng=rng), fontname="Roboto-Bold", fontsize=80, filling="KTH_cotton", base_path=texture_path, rng=rng)
        image4, mask4 = get_fill("KTH_orange_peel", image_size, base_path=texture_path, rng=rng)
        image, masks = compose((image1, mask1),(image2, mask2),(image3, mask3),(image4, mask4))
        save(image, f"img_{label}_{i}", label, masks, basefolder = savepath, ground_truth = mask2)


# create A-B plus dataset
def create_ABplus(savepath = r"./data/datasets/ABplus", image_size=224, texture_path = r"./data/textures/train", seed=0):
    Path(savepath).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(200):
        # create A
        label = "A"
        image1, mask1 = draw_text("A", image_size, position=get_position(image_size, "", item_size=40, rng=rng), fontname="Roboto-Bold", fontsize=40, filling="KTH_aluminium_foil", base_path=texture_path, rng=rng)
        image2, mask2 = draw_text("", image_size, position=get_position(image_size, "", item_size=40, rng=rng), fontname="Roboto-Bold", fontsize=40, filling="G", base_path=texture_path, rng=rng)
        image3, mask3 = draw_text("+", image_size, position=get_position(image_size, "", item_size=80, rng=rng), fontname="Roboto-Bold", fontsize=80, filling="B", base_path=texture_path, rng=rng)
        image4, mask4 = draw_text("*", image_size, position=get_position(image_size, "", item_size=80, rng=rng), fontname="Roboto-Bold", fontsize=80, filling="R", base_path=texture_path, rng=rng)
        image5, mask5 = draw_text("/", image_size, position=get_position(image_size, "", item_size=80, rng=rng), fontname="Roboto-Bold", fontsize=80, filling="Y", base_path=texture_path, rng=rng)
        image6, mask6 = draw_text("#", image_size, position=get_position(image_size, "", item_size=80, rng=rng), fontname="Roboto-Bold", fontsize=80, filling="Y", base_path=texture_path, rng=rng)
        image7, mask7 = draw_text("X", image_size, position=get_position(image_size, "", item_size=80, rng=rng), fontname="Roboto-Bold", fontsize=80, filling="K", base_path=texture_path, rng=rng)
        image8, mask8 = get_fill("KTH_sponge", image_size, base_path=texture_path, rng=rng)
        image, masks = compose((image1, mask1),(image2, mask2),(image3, mask3),(image4, mask4),(image5, mask5),(image6, mask6),(image7, mask7),(image8, mask8))
        save(image, f"img_{label}_{i}", label, masks, basefolder = savepath, ground_truth = mask1)
        # create B
        label = "B"
        image1, mask1 = draw_text("", image_size, position=get_position(image_size, "", item_size=40, rng=rng), fontname="Roboto-Bold", fontsize=40, filling="G", base_path="", rng=rng)
        image2, mask2 = draw_text("B", image_size, position=get_position(image_size, "", item_size=40, rng=rng), fontname="Roboto-Bold", fontsize=40, filling="G", base_path="", rng=rng)
        image3, mask3 = draw_text("+", image_size, position=get_position(image_size, "", item_size=80, rng=rng), fontname="Roboto-Bold", fontsize=80, filling="B", base_path=texture_path, rng=rng)
        image4, mask4 = draw_text("*", image_size, position=get_position(image_size, "", item_size=80, rng=rng), fontname="Roboto-Bold", fontsize=80, filling="R", base_path=texture_path, rng=rng)
        image5, mask5 = draw_text("/", image_size, position=get_position(image_size, "", item_size=80, rng=rng), fontname="Roboto-Bold", fontsize=80, filling="Y", base_path=texture_path, rng=rng)
        image6, mask6 = draw_text("#", image_size, position=get_position(image_size, "", item_size=80, rng=rng), fontname="Roboto-Bold", fontsize=80, filling="Y", base_path=texture_path, rng=rng)
        image7, mask7 = draw_text("X", image_size, position=get_position(image_size, "", item_size=80, rng=rng), fontname="Roboto-Bold", fontsize=80, filling="K", base_path=texture_path, rng=rng)
        image8, mask8 = get_fill("KTH_sponge", image_size, base_path=texture_path, rng=rng)
        image, masks = compose((image1, mask1),(image2, mask2),(image3, mask3),(image4, mask4),(image5, mask5),(image6, mask6),(image7, mask7),(image8, mask8))
        save(image, f"img_{label}_{i}", label, masks, basefolder = savepath, ground_truth = mask2)

# create C-O dataset
def create_CO(savepath = r"./data/datasets/CO", image_size=224, texture_path = r"./data/textures/train", seed=0):
    Path(savepath).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(200):
        # create C
        label = "C"
        image1, mask1 = draw_text("C", image_size, position=get_position(image_size, "", item_size=100, rng=rng), fontname="Roboto-Bold", fontsize=100, filling="KTH_aluminium_foil", base_path=texture_path, rng=rng)
        image2, mask2 = draw_text("", image_size, position=get_position(image_size, "", item_size=100, rng=rng), fontname="Roboto-Bold", fontsize=100, filling="KTH_aluminium_foil", base_path=texture_path, rng=rng)
        image3, mask3 = draw_text("+", image_size, position=get_position(image_size, "", item_size=80, rng=rng), fontname="Roboto-Bold", fontsize=80, filling="KTH_cotton", base_path=texture_path, rng=rng)
        image4, mask4 = get_fill("KTH_cork", image_size, base_path=texture_path, rng=rng)
        image, masks = compose((image1, mask1),(image2, mask2),(image3, mask3),(image4, mask4))
        save(image, f"img_{label}_{i}", label, masks, basefolder = savepath, ground_truth = mask1)
        # create O
        label = "O"
        image1, mask1 = draw_text("", image_size, position=get_position(image_size, "", item_size=100, rng=rng), fontname="Roboto-Bold", fontsize=100, filling="KTH_aluminium_foil", base_path=texture_path, rng=rng)
        image2, mask2 = draw_text("O", image_size, position=get_position(image_size, "", item_size=100, rng=rng), fontname="Roboto-Bold", fontsize=100, filling="KTH_aluminium_foil", base_path=texture_path, rng=rng)
        image3, mask3 = draw_text("+", image_size, position=get_position(image_size, "", item_size=80, rng=rng), fontname="Roboto-Bold", fontsize=80, filling="KTH_cotton", base_path=texture_path, rng=rng)
        image4, mask4 = get_fill("KTH_cork", image_size, base_path=texture_path, rng=rng)
        image, masks = compose((image1, mask1),(image2, mask2),(image3, mask3),(image4, mask4))
        save(image, f"img_{label}_{i}", label, masks, basefolder = savepath, ground_truth = mask2)

# create big-small dataset
def create_BigSmall(savepath = r"./data/datasets/BigSmall", image_size=224, texture_path = r"./data/textures/train", seed=0):
    Path(savepath).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(200):
        # create Big
        label = "big"
        image1, mask1 = draw_text("B", image_size, position=get_position(image_size, "", item_size=100), fontname="Roboto-Bold", fontsize=100, filling="B", base_path=texture_path, rng=rng)
        image2, mask2 = draw_text("", image_size, position=get_position(image_size, "", item_size=40), fontname="Roboto-Bold", fontsize=40, filling="B", base_path=texture_path, rng=rng)
        image3, mask3 = draw_text("+", image_size, position=get_position(image_size, "", item_size=80), fontname="Roboto-Bold", fontsize=80, filling="KTH_cotton", base_path=texture_path, rng=rng)
        image4, mask4 = get_fill("KTH_cork", image_size, base_path=texture_path, rng=rng)
        image, masks = compose((image1, mask1),(image2, mask2),(image3, mask3),(image4, mask4))
        save(image, f"img_{label}_{i}", label, masks, basefolder = savepath, ground_truth = mask1)
        # create small
        label = "small"
        image1, mask1 = draw_text("", image_size, position=get_position(image_size, "", item_size=100, rng=rng), fontname="Roboto-Bold", fontsize=100, filling="B", base_path="", rng=rng)
        image2, mask2 = draw_text("B", image_size, position=get_position(image_size, "", item_size=40, rng=rng), fontname="Roboto-Bold", fontsize=40, filling="B", base_path="", rng=rng)
        image3, mask3 = draw_text("+", image_size, position=get_position(image_size, "", item_size=80, rng=rng), fontname="Roboto-Bold", fontsize=80, filling="KTH_cotton", base_path=texture_path, rng=rng)
        image4, mask4 = get_fill("KTH_cork", image_size, base_path=texture_path, rng=rng)
        image, masks = compose((image1, mask1),(image2, mask2),(image3, mask3),(image4, mask4))
        save(image, f"img_{label}_{i}", label, masks, basefolder = savepath, ground_truth = mask2)

# create color dataset
def create_colorGB(savepath = r"./data/datasets/colorGB", image_size=224, texture_path = r"./data/textures/train", seed=0):
    Path(savepath).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(400):
        letter1 = rng.choice(["A", "B"])
        letter2 = rng.choice(["C", "D", ""])
        # create 
        label = rng.choice(["G", "B"])
        image1, mask1 = draw_text(letter1, image_size, position=get_position(image_size, "", item_size=40, rng=rng), fontname="Roboto-Bold", fontsize=40, filling=label, base_path="", rng=rng)
        image2, mask2 = draw_text(letter2, image_size, position=get_position(image_size, "", item_size=40, rng=rng), fontname="Roboto-Bold", fontsize=40, filling="G", base_path="", rng=rng)
        image3, mask3 = draw_text("+", image_size, position=get_position(image_size, "", item_size=80, rng=rng), fontname="Roboto-Bold", fontsize=80, filling="KTH_cotton", base_path=texture_path, rng=rng)
        image4, mask4 = get_fill("KTH_orange_peel", image_size, base_path=texture_path, rng=rng)
        image, masks = compose((image1, mask1),(image2, mask2),(image3, mask3),(image4, mask4))
        save(image, f"img_{label}_{i}", label, masks, basefolder = savepath, ground_truth = mask1)

# create A-NA dataset
def create_ANA(savepath = r"./data/datasets/ANA", image_size=224, texture_path = r"./data/textures/train", seed=0):
    Path(savepath).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(200):
        # create A
        label = "A"
        image1, mask1 = draw_text("A", image_size, position=get_position(image_size, "", item_size=40), fontname="Roboto-Bold", fontsize=40, filling="B", base_path=texture_path, rng=rng)
        image4, mask4 = get_fill("W", image_size, base_path=texture_path, rng=rng)
        image, masks = compose((image1, mask1),(image4, mask4))
        save(image, f"img_{label}_{i}", label, masks, basefolder = savepath, ground_truth = mask1)
        # create NA
        label = "NA"
        image1, mask1 = draw_text("", image_size, position=get_position(image_size, "", item_size=40), fontname="Roboto-Bold", fontsize=40, filling="B", base_path=texture_path, rng=rng)
        image4, mask4 = get_fill("W", image_size, base_path=texture_path, rng=rng)
        image, masks = compose((image1, mask1),(image4, mask4))
        save(image, f"img_{label}_{i}", label, masks, basefolder = savepath, ground_truth = mask1)

# create Up-Down dataset
def create_UpDown(savepath = r"./data/datasets/UpDown", image_size=224, texture_path = r"./data/textures/train", seed=0):
    Path(savepath).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(200):
        # create Up
        label = "up"
        image1, mask1 = draw_text("A", image_size, position=get_position(image_size, "top", item_size=40), fontname="Roboto-Bold", fontsize=40, filling="B", base_path=texture_path, rng=rng)
        image2, mask2 = draw_text("", image_size, position=get_position(image_size, "bottom", item_size=40), fontname="Roboto-Bold", fontsize=40, filling="B", base_path=texture_path, rng=rng)
        image4, mask4 = get_fill("gray", image_size, base_path=texture_path, rng=rng)
        image, masks = compose((image1, mask1),(image2, mask2),(image4, mask4))
        save(image, f"img_{label}_{i}", label, masks, basefolder = savepath, ground_truth = mask1)
        # create Down
        label = "down"
        image1, mask1 = draw_text("", image_size, position=get_position(image_size, "top", item_size=40), fontname="Roboto-Bold", fontsize=40, filling="B", base_path=texture_path, rng=rng)
        image2, mask2 = draw_text("A", image_size, position=get_position(image_size, "bottom", item_size=40), fontname="Roboto-Bold", fontsize=40, filling="B", base_path=texture_path, rng=rng)
        image4, mask4 = get_fill("gray", image_size, base_path=texture_path, rng=rng)
        image, masks = compose((image1, mask1),(image2, mask2),(image4, mask4))
        save(image, f"img_{label}_{i}", label, masks, basefolder = savepath, ground_truth = mask2)

# create Ap-Down dataset
def create_ApDown(savepath = r"./data/datasets/ApDown", image_size=224, texture_path = r"./data/textures/train", seed=0):
    Path(savepath).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(200):
        # create Up
        label = "ap"
        image1, mask1 = draw_text("A", image_size, position=get_position(image_size, "top", item_size=40), fontname="Roboto-Bold", fontsize=40, filling="B", base_path=texture_path, rng=rng)
        image2, mask2 = draw_text("", image_size, position=get_position(image_size, "bottom", item_size=40), fontname="Roboto-Bold", fontsize=40, filling="B", base_path=texture_path, rng=rng)
        image4, mask4 = get_fill("gray", image_size, base_path=texture_path, rng=rng)
        image, masks = compose((image1, mask1),(image2, mask2),(image4, mask4))
        save(image, f"img_{label}_{i}", label, masks, basefolder = savepath, ground_truth = mask1)
        # create Down
        label = "down"
        image1, mask1 = draw_text("", image_size, position=get_position(image_size, "top", item_size=40), fontname="Roboto-Bold", fontsize=40, filling="B", base_path=texture_path, rng=rng)
        image2, mask2 = draw_text("D", image_size, position=get_position(image_size, "bottom", item_size=40), fontname="Roboto-Bold", fontsize=40, filling="B", base_path=texture_path, rng=rng)
        image4, mask4 = get_fill("gray", image_size, base_path=texture_path, rng=rng)
        image, masks = compose((image1, mask1),(image2, mask2),(image4, mask4))
        save(image, f"img_{label}_{i}", label, masks, basefolder = savepath, ground_truth = mask2)

# create isA dataset
def create_isA(savepath = r"./data/datasets/isA", image_size=224, texture_path = r"./data/textures/train", seed=0):
    Path(savepath).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(200):
        # create isA
        label = "isA"
        image1, mask1 = draw_text("A", image_size, position=get_position(image_size, "", item_size=120), fontname="Roboto-Bold", fontsize=120, filling="B", base_path=texture_path, rng=rng)
        image2, mask2 = draw_text("", image_size, position=get_position(image_size, "", item_size=120), fontname="Roboto-Bold", fontsize=120, filling="B", base_path=texture_path, rng=rng)
        image4, mask4 = get_fill("gray", image_size, base_path=texture_path, rng=rng)
        image, masks = compose((image1, mask1),(image2, mask2),(image4, mask4))
        save(image, f"img_{label}_{i}", label, masks, basefolder = savepath, ground_truth = mask1)
        # create notA
        label = "notA"
        letter = rng.choice(list("BCDEFGH"))
        image1, mask1 = draw_text("", image_size, position=get_position(image_size, "", item_size=120, rng=rng), fontname="Roboto-Bold", fontsize=120, filling="B", base_path=texture_path, rng=rng)
        image2, mask2 = draw_text(letter, image_size, position=get_position(image_size, "", item_size=120, rng=rng), fontname="Roboto-Bold", fontsize=120, filling="B", base_path=texture_path, rng=rng)
        image4, mask4 = get_fill("gray", image_size, base_path=texture_path, rng=rng)
        image, masks = compose((image1, mask1),(image2, mask2),(image4, mask4))
        save(image, f"img_{label}_{i}", label, masks, basefolder = savepath, ground_truth = mask2)