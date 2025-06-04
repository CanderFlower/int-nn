# Load necessary package
library(grid)

# Function to draw a layer (box with nodes and label)
draw_layer <- function(x_center, y_center, label, num_nodes,
                       box_width = 0.2, box_height = 0.055, # Adjusted box_height
                       node_radius_npc = 0.012,
                       label_text_y_offset_factor = 0.28, # Offset for label within box
                       box_lwd = 2, node_lwd = 1.5,
                       box_r = 0.01, label_cex = 1.0) {

    # Draw the rounded rectangle for the layer
    grid.roundrect(x = unit(x_center, "npc"), y = unit(y_center, "npc"),
                   width = unit(box_width, "npc"), height = unit(box_height, "npc"),
                   r = unit(box_r, "snpc"),
                   gp = gpar(lwd = box_lwd, fill = "white"))

    # Add layer label (positioned higher within the box)
    grid.text(label, x = unit(x_center, "npc"), y = unit(y_center + box_height * label_text_y_offset_factor, "npc"),
              just = "center", gp = gpar(fontface = "plain", cex = label_cex))

    # Draw nodes (circles)
    if (num_nodes > 0) {
        node_y_pos = y_center - box_height * 0.1 # Position nodes slightly lower in box
        if (num_nodes == 1) {
             node_x_coords <- x_center
        } else {
            node_x_coords <- seq(x_center - box_width / 2 + box_width * 0.18, # Adjusted padding
                                 x_center + box_width / 2 - box_width * 0.18,
                                 length.out = num_nodes)
        }
        
        for (nx in node_x_coords) {
            grid.circle(x = unit(nx, "npc"), y = unit(node_y_pos, "npc"),
                        r = unit(node_radius_npc, "npc"),
                        gp = gpar(fill = "grey80", col = "black", lwd = node_lwd))
        }
    }
    # Return coordinates for arrow connections
    return(list(
        top = y_center + box_height / 2,
        bottom = y_center - box_height / 2,
        left = x_center - box_width / 2,
        right = x_center + box_width / 2,
        x_center = x_center,
        y_center = y_center
    ))
}

# --- Main plotting ---
grid.newpage()

# --- Parameters ---
x_center_network <- 0.5
box_w <- 0.22 # Slightly wider boxes
box_h <- 0.055
node_r_npc <- 0.011 # Slightly smaller nodes to fit more
layer_label_cex <- 0.9 # Smaller font for layer labels

# Layer y-positions (bottom to top) - adjusted for even spacing
y_base <- 0.08
y_spacing <- 0.125 # Increased spacing
y_input <- y_base
y_h1    <- y_base + 1 * y_spacing
y_h2    <- y_base + 2 * y_spacing
y_h3    <- y_base + 3 * y_spacing
y_h4    <- y_base + 4 * y_spacing
y_h5    <- y_base + 5 * y_spacing
y_output<- y_base + 6 * y_spacing


# Arrow parameters
arrow_lwd <- 2
arrow_col <- "black"
arrow_type <- arrow(angle = 20, length = unit(0.08, "inches"), type = "closed")
feedback_arrow_offset_factor <- 0.4 # How far out the feedback arrows go, relative to box_w/2
feedback_x_bend_left <- x_center_network - box_w/2 - box_w * feedback_arrow_offset_factor
feedback_x_bend_right <- x_center_network + box_w/2 + box_w * feedback_arrow_offset_factor


# --- Draw Layers ---
coords_input <- draw_layer(x_center_network, y_input, "Input layer", 2, box_w, box_h, node_r_npc, label_cex=layer_label_cex)
coords_h1    <- draw_layer(x_center_network, y_h1, "Hidden layer 1", 3, box_w, box_h, node_r_npc, label_cex=layer_label_cex)
coords_h2    <- draw_layer(x_center_network, y_h2, "Hidden layer 2", 4, box_w, box_h, node_r_npc, label_cex=layer_label_cex)
coords_h3    <- draw_layer(x_center_network, y_h3, "Hidden layer 3", 4, box_w, box_h, node_r_npc, label_cex=layer_label_cex)
coords_h4    <- draw_layer(x_center_network, y_h4, "Hidden layer 4", 4, box_w, box_h, node_r_npc, label_cex=layer_label_cex)
coords_h5    <- draw_layer(x_center_network, y_h5, "Hidden layer 5", 3, box_w, box_h, node_r_npc, label_cex=layer_label_cex)
coords_output<- draw_layer(x_center_network, y_output, "Output layer", 1, box_w, box_h, node_r_npc, label_cex=layer_label_cex)

all_coords <- list(coords_input, coords_h1, coords_h2, coords_h3, coords_h4, coords_h5, coords_output)

# --- Draw Feed-forward Arrows (Bottom-up) ---
for (i in 1:(length(all_coords) - 1)) {
    from_layer <- all_coords[[i]]
    to_layer   <- all_coords[[i+1]]
    grid.segments(x0 = unit(from_layer$x_center, "npc"), y0 = unit(from_layer$top, "npc"),
                  x1 = unit(to_layer$x_center, "npc"),   y1 = unit(to_layer$bottom, "npc"),
                  gp = gpar(lwd = arrow_lwd, col = arrow_col), arrow = arrow_type)
}

# --- Draw Feedback Arrows (Top-down, around the boxes) ---
layers_for_feedback <- list(coords_output, coords_h5, coords_h4, coords_h3, coords_h2, coords_h1)

# Left side feedback path
for (i in 1:(length(layers_for_feedback) - 1)) {
    from_l <- layers_for_feedback[[i]]
    to_l   <- layers_for_feedback[[i+1]]
    grid.lines(x = unit(c(from_l$left, feedback_x_bend_left, feedback_x_bend_left, to_l$left), "npc"),
               y = unit(c(from_l$y_center-0.005, from_l$y_center-0.005, to_l$y_center+0.005, to_l$y_center+0.005), "npc"),
               gp = gpar(lwd = arrow_lwd, col = arrow_col), arrow = arrow_type)
}

# Right side feedback path
for (i in 1:(length(layers_for_feedback) - 1)) {
    from_l <- layers_for_feedback[[1]]
    to_l   <- layers_for_feedback[[i+1]]
    grid.lines(x = unit(c(from_l$right, feedback_x_bend_right, feedback_x_bend_right, to_l$right), "npc"),
               y = unit(c(from_l$y_center, from_l$y_center, to_l$y_center, to_l$y_center), "npc"),
               gp = gpar(lwd = arrow_lwd, col = arrow_col), arrow = arrow_type)
}


# --- Left Column: Integer BP ---
x_left_col <- 0.20 # Adjusted for wider network
y_title_main_offset <- 0.04
y_text_spacing <- 0.028
text_cex <- 0.9 # Base cex for text
title_cex <- 1.2

grid.text("BP", x = unit(x_left_col, "npc"), y = unit(y_output + y_title_main_offset + 0.05, "npc"),
          gp = gpar(fontface = "bold", cex = title_cex))
grid.text("Error: e bits", x = unit(x_left_col, "npc"), y = unit(y_output + y_title_main_offset + y_text_spacing*0.5, "npc"), just = "center", gp = gpar(cex = text_cex))
grid.text("Weight: w bits", x = unit(x_left_col, "npc"), y = unit(y_output + y_title_main_offset - y_text_spacing*0.5, "npc"), just = "center", gp = gpar(cex = text_cex))
# grid.text(expression(delta[BP]^"[k]" ~ "grows exponentially"),
#           x = unit(x_left_col, "npc"), y = unit(y_output + y_title_main_offset - y_text_spacing*2.2, "npc"), just = "center", gp = gpar(cex = text_cex))

# Delta BP formulas
bp_y_coords <- c(coords_h5$y_center, coords_h4$y_center, coords_h3$y_center, coords_h2$y_center, coords_h1$y_center)
bp_superscripts <- 5:1
bp_w_multipliers <- 1:5

for (i in 1:5) {
    k_val <- bp_superscripts[i]
    w_mult <- bp_w_multipliers[i]
    if (w_mult == 1) {
      formula_bp <- bquote(delta[BP]^{"["*.(k_val)*"]"} == O(2^{e+w}))
    } else {
      formula_bp <- bquote(delta[BP]^{"["*.(k_val)*"]"} == O(2^{e+.(w_mult)*w}))
    }
    grid.text(formula_bp, x = unit(x_left_col, "npc"), y = unit(bp_y_coords[i], "npc"), just = "center", gp = gpar(cex = text_cex))
}



# --- Right Column: Integer DFA ---
x_right_col <- 1 - x_left_col # Symmetrical

grid.text("DFA", x = unit(x_right_col, "npc"), y = unit(y_output + y_title_main_offset + 0.05, "npc"),
          gp = gpar(fontface = "bold", cex = title_cex))
grid.text("Error: e bits", x = unit(x_right_col, "npc"), y = unit(y_output + y_title_main_offset + y_text_spacing*0.5, "npc"), just = "center", gp = gpar(cex = text_cex))
grid.text("Random weight: r bits", x = unit(x_right_col, "npc"), y = unit(y_output + y_title_main_offset - y_text_spacing*0.5, "npc"), just = "center", gp = gpar(cex = text_cex))
# grid.text(expression(delta[DFA]^"[k]" ~ "does not grow"),
#           x = unit(x_right_col, "npc"), y = unit(y_output + y_title_main_offset - y_text_spacing*2.2, "npc"), just = "center", gp = gpar(cex = text_cex))

# Delta DFA formulas
dfa_y_coords <- bp_y_coords # Same y-alignment
dfa_superscripts <- 5:1

for (i in 1:5) {
    k_val <- dfa_superscripts[i]
    formula_dfa <- bquote(delta[DFA]^{"["*.(k_val)*"]"} == O(2^{e+r}))
    grid.text(formula_dfa, x = unit(x_right_col, "npc"), y = unit(dfa_y_coords[i], "npc"), just = "center", gp = gpar(cex = text_cex))
}