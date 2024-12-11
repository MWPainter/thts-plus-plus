"""
Code to read in the moves made in a game of go, and print images of the board after every move, including the areas
used for the scoring.

N.B. need to make sure that "external/katago/python" is in the $PYTHONPATH bash variable for this code to work.
"""

import cairo
import numpy as np
import os
import sys

from external.KataGo.python.board import Board


def _print_state_to_png(filename, board_array, board_size, loc_x_fn, loc_y_fn):
    """Takes an array of stones and prints a nice png image of the board. """
    svg_filename_parts = filename.split(".")
    svg_filename_parts[-1] = "svg"
    svg_filename = ".".join(svg_filename_parts)

    img_size = 100 * board_size

    with cairo.SVGSurface(svg_filename, img_size, img_size) as surface:
        ctx = cairo.Context(surface)
        scale = 100.0
        ctx.scale(scale, scale)

        ctx.save()
        ctx.set_source_rgb(164.0/255.0, 116.0/255.0, 73.0/255.0)
        ctx.paint()
        ctx.restore()

        for x in range(board_size - 1):
            for y in range(board_size - 1):
                top_left = (x+0.5, y+0.5)
                _plot_square(ctx=ctx, top_left=top_left)

        for i in range(board_array.shape[0]):
            if board_array[i] in {Board.BLACK, Board.WHITE}:
                x, y = loc_x_fn(loc=i), loc_y_fn(loc=i)
                center = (x+0.5, y+0.5)
                player = board_array[i]
                if player == Board.BLACK:
                    _plot_black_circle(ctx=ctx, center=center)
                elif player == Board.WHITE:
                    _plot_white_circle(ctx=ctx, center=center)

        surface.write_to_png(filename)

def _plot_square(ctx, top_left, width=1.0):
    """Cairo helper. """
    ctx.set_source_rgb(0.0, 0.0, 0.0)
    ctx.set_line_width(0.06)
    ctx.rectangle(top_left[0], top_left[1], width, width)
    ctx.stroke()

def _plot_white_circle(ctx, center, radius=0.5):
    """Cairo helper. """
    ctx.set_source_rgb(1.0, 1.0, 1.0)
    ctx.set_line_width(0.06)
    ctx.arc(center[0], center[1], radius, 0, 2.0*np.pi)
    ctx.fill()

def _plot_black_circle(ctx, center, radius=0.5):
    """Cairo helper. """
    ctx.set_source_rgb(0.0, 0.0, 0.0)
    ctx.set_line_width(0.06)
    ctx.arc(center[0], center[1], radius, 0, 2.0*np.pi)
    ctx.fill()



def create_directory_for_images(match_csv_filename):
    """Makes a folder for the images we're about to produce, and returns the string for the directory."""
    game_imgs_dir = "{base}_game_imgs".format(base=match_csv_filename[:-4])
    if not os.path.exists(game_imgs_dir):
        os.makedirs(game_imgs_dir)
    else:
        raise Exception("Img folder already exists for match, delete folder if want to redo. Skipping for now.")
    return game_imgs_dir

def read_moves_from_moves_file(match_csv_filename):
    """Takes the filename of a match csv and reads the moves from it."""
    # Read in the file and remove the header from it, with a sanity check
    file_lines = []
    with open(match_csv_filename) as match_file:
        file_lines = match_file.readlines()
    if file_lines[1].split(",")[2] != "pos":
        raise Exception("Match file not in format expected")
    file_lines = file_lines[2:]

    # Parse each of the lines for a move. Of the form (x|y), so need to strip brackets and convert to ints
    moves = []
    for line in file_lines:
        move_str = line.split(",")[2]
        x_str, y_str = move_str[1:-1].split("|")
        moves.append((int(x_str), int(y_str)))
    
    return moves

def make_board_and_area_imgs(board, board_img_name, area_img_name):
    area = np.array(np.zeros(board.board.shape[0]))
    area -= 1.0
    nonPassAliveStones = False
    safeBigTerritories = True
    unsafeBigTerritories = False
    isMultiStoneSuicideLegal = False
    board.calculateArea(
        result=area, 
        nonPassAliveStones=nonPassAliveStones, 
        safeBigTerritories=safeBigTerritories, 
        unsafeBigTerritories=unsafeBigTerritories, 
        isMultiStoneSuicideLegal=isMultiStoneSuicideLegal)

    _print_state_to_png(
        filename=board_img_name, 
        board_array=board.board, 
        board_size=board.size, 
        loc_x_fn=board.loc_x, 
        loc_y_fn=board.loc_y)
    _print_state_to_png(
        filename=area_img_name, 
        board_array=area, 
        board_size=board.size, 
        loc_x_fn=board.loc_x, 
        loc_y_fn=board.loc_y)
    
def make_game_imgs(imgs_dir, moves, board_size):
    """Uses the list of moves to produce images of the board state and the area scoring"""
    board_img_format_str = os.path.join(imgs_dir, "move_{move}.png")
    area_img_format_str = os.path.join(imgs_dir, "scoring_area_{move}.png")

    board = Board(size=board_size)
    board_img_str = board_img_format_str.format(move=0)
    area_img_str = area_img_format_str.format(move=0)
    make_board_and_area_imgs(board=board, board_img_name=board_img_str, area_img_name=area_img_str)

    for i,(x,y) in enumerate(moves):
        loc = board.PASS_LOC
        if x != -1 and y != -1:
            loc = board.loc(x,y)
        board.play(pla=board.pla, loc=loc)
        board_img_str = board_img_format_str.format(move=i+1)
        area_img_str = area_img_format_str.format(move=i+1)
        make_board_and_area_imgs(board=board, board_img_name=board_img_str, area_img_name=area_img_str)


if __name__ == "__main__":
    # get the args
    if len(sys.argv) != 4:
        raise Exception("Expected usage: python produce_go_graphics.py <board_size> <csv_depth> <base_dir>")
    
    board_size = int(sys.argv[1])
    csv_depth = int(sys.argv[2])
    base_dir = sys.argv[3]

    # get all top level directories to check for match csvs
    candidate_dirs = [base_dir]
    while csv_depth > 0:
        new_candidate_dirs = []
        for candidate_dir in candidate_dirs:
            if os.path.isdir(candidate_dir):
                print("Searching dir: {dir}".format(dir=candidate_dir))
                for filename in os.listdir(candidate_dir):
                    if "trees" in filename:
                        continue
                    new_candidate_dirs.append(os.path.join(candidate_dir,filename))
        candidate_dirs = new_candidate_dirs
        csv_depth -= 1

    # get all the match csvs from the top level directories
    match_csvs = []
    for candidate_dir in candidate_dirs:
        print("Looking in {dir} for match csvs".format(dir=candidate_dir))
        if os.path.isdir(candidate_dir):
            for filename in os.listdir(candidate_dir):
                if "match" in filename and ".csv" in filename:
                    match_csvs.append(os.path.join(candidate_dir,filename))
    
    # print out that we did our traversing and how many games we're doing to print images for
    print("Found {n} matches to make graphics for".format(n=len(match_csvs)))

    # Make graphics for each match csv
    for match_csv_filename in match_csvs:
        print("Making graphics for match file: {filename}".format(filename=match_csv_filename))
        try:
            imgs_dir = create_directory_for_images(match_csv_filename=match_csv_filename)
            moves = read_moves_from_moves_file(match_csv_filename=match_csv_filename)
            make_game_imgs(imgs_dir=imgs_dir, moves=moves, board_size=board_size)
        except Exception as e:
            print("An error occurred with this one:")
            print(e)
