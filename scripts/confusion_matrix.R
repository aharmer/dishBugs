library(tidyverse)

confusion_df = read_csv("D:/Dropbox/data/dishBugs/runs/classify/val2/confusion_matrix.csv")

# Assuming your first column is the actual classes
# and the remaining columns are the predicted classes
confusion_long <- confusion_df %>%
  pivot_longer(-1, names_to = "actual", values_to = "count")

# If you need to rename the first column
confusion_long <- confusion_df %>%
  rename(actual = 1) %>%  # Rename first column to 'actual'
  pivot_longer(-actual, names_to = "predicted", values_to = "count")

confusion_long[confusion_long == 0] = NA

confusion_long = confusion_long %>% 
  mutate(actual = str_to_title(actual)) %>% 
  mutate(predicted = str_to_title(predicted))


ggplot(confusion_long, aes(x = predicted, y = actual, fill = count)) +
  geom_tile(colour = "white", size = 0) +
  geom_text(aes(label = count), color = "black", size = 3.5) +
  scale_fill_gradient(
    low = "#fefed7",
    high = "#2899c1",
    space = "Lab",
    na.value = "white",
    guide = "none",
    transform = "log10",
    aesthetics = "fill") +
  scale_y_discrete(limits = rev) +  # Reverse y-axis for conventional matrix view
  labs(
    x = "Actual Class",
    y = "Predicted Class\n") +
  theme_minimal() +
  theme(
   axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text.y = element_text(hjust = 1),
    panel.grid = element_blank())


