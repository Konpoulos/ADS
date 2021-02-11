#question 1

set.seed(1234)
student_grade  <- rnorm(32, 7)
student_number <- round(runif(32) * 2e6 + 5e6)
programme      <- sample(c("Science", "Social Science"), 32, replace = TRUE)

#Create the Data Frame
gg_students <- data.frame(student_grade = (student_grade), 
                          student_number = as.character(student_number), 
                          programme = as.character(programme),
                          stringsAsFactors = FALSE)


#3 Question
ggplot(Hitters,aes(x=HmRun,y=Hits))+
  geom_point()+
  labs(y="Hits",x="Home Runs")

#4 Question

homeruns_plot2 <- 
  ggplot(Hitters, aes(x = HmRun, y = Hits, colour = League, size = Salary)) +
  geom_point() +
  labs(x = "HmRun", y = "Hits")

homeruns_plot2

#6 Question

gg_students_2 <- 
  ggplot(gg_students,aes(x=student_grade)) +
  geom_histogram(binwidth = NULL)

gg_students_2

#7 Question

gg_students_3 <- 
  ggplot(gg_students,aes(x=student_grade)) +
  geom_density(fill = "light seagreen")

gg_students_3

#8 Question
gg_students_3 <- 
  ggplot(gg_students,aes(x=student_grade)) +
  geom_density(fill = "light seagreen") +
  geom_rug(colour="green",size=3)

gg_students_3

#9 Question
gg_students_4 <- 
  ggplot(gg_students,aes(x=student_grade)) +
  geom_density(fill = "light seagreen",outline.type = "lower") +
  geom_rug(colour="green",size=3) +
  labs(y = NULL) +
  theme_minimal() +
  xlim(0,10)

gg_students_4

#10 Question
gg_students_box <-
  ggplot(gg_students,aes(x=programme,y=student_grade,fill=programme))+
  geom_boxplot()

gg_students_box

#11 Question
#What do each of the horizontal lines in the boxplot mean? What do the vertical lines (whiskers) mean?
  
#  Answer:
 # Summary statistics
#The lower and upper hinges correspond to the first and third quartiles (the 25th and 75th percentiles). This differs slightly from the method used by the boxplot() function, and may be apparent with small samples. See boxplot.stats() for for more information on how hinge positions are calculated for boxplot().

#The upper whisker extends from the hinge to the largest value no further than 1.5 * IQR from the hinge (where IQR is the inter-quartile range, or distance between the first and third quartiles). The lower whisker extends from the hinge to the smallest value at most 1.5 * IQR of the hinge. Data beyond the end of the whiskers are called "outlying" points and are plotted individually.

# In a notched box plot, the notches extend 1.58 * IQR / sqrt(n). This gives a roughly 95% confidence interval for comparing medians. See McGill et al. (1978) for more details

#12 Question 
gg_students_dens <- 
  ggplot(gg_students,aes(x=student_grade,colour=programme)) +
  geom_density(fill = "light seagreen",alpha = 0.1)+
  geom_rug(colour="green",size=3)

gg_students_dens

#13 Question 
bar_plot <- 
  ggplot(Hitters, aes(Years))+
  stat_count()+
  geom_bar()
bar_plot

#14 Question

#create the 200 apperances
first200 <- select(Smarket[1:200,],Volume)

#Create the mutate x-positions
firsts200 <- mutate(first200, days=1:200)

line_plot <- 
  ggplot(firsts200,aes(x=days,y=Volume))+
  geom_line()
line_plot

#15 
line_plot <- 
  ggplot(firsts200,aes(x=days,y=Volume))+
  geom_line(colour="green")+
  geom_point(colour="green")

line_plot

#16 Question

whichday <- which.max(firsts200$Volume)

whichvol <- max(firsts200$Volume)

#17 Question need the 16 too
label_plot <- 
  ggplot(firsts200,aes(x=days,y=Volume))+
  geom_point()+
  geom_label(aes(x = whichday, y = whichvol, label = "Peak volume"))
label_plot

#18 Question
baseball <- Hitters
baseball <- filter(baseball, !is.na(Salary))
baseball <- mutate(baseball,Salary_Range=cut(baseball$Salary,breaks=3,labels = c("low","mid","high")))
baseball <- mutate(baseball,Prop=HmRun/Hits)
baseballFilter <- select(baseball,Salary,Salary_Range,Hits,HmRun,Prop)
head(baseballFilter)

#19 Question
baseball_plot <- 
  ggplot(baseball,aes(x=CWalks,y=Prop))+
  geom_point(color="green")+
  xlim(0,1600)+
  ylim(0,0.4)+
  labs(y="home proportion",title="Home runs x Cwalks")

baseball_plot

#20 Question
baseball_plot <- 
  ggplot(baseball,aes(x=CWalks,y=Prop))+
  geom_point(color="green")+
  xlim(0,1600)+
  ylim(0,0.4)+
  labs(y="home proportion",title="Home runs x Cwalks")+
  facet_wrap(~Salary_Range)
  
  baseball_plot
  
  
  
  
homeruns_plot2 <- 
    ggplot(Hitters, aes(x = HmRun, y = Hits)) +
    geom_point(aes(colour = League)) +
    labs(x = "HmRun", y = "Hits")
  
  homeruns_plot2
  
  
  
  homeruns_plot2 <- 
    ggplot(Hitters, aes(x = HmRun, y = Hits)) +
    geom_point(colour = League) +
    labs(x = "HmRun", y = "Hits")
  
  homeruns_plot2  
  

ggplot(trees,aes(x=Girth,y=Height,size=Volume))+
  geom_point()

trees%>%
  ggplot(aes(x=Girth,y=Height))+
  geom_point(aes(size=Volume))
  
library(tidyverse)  
Hitters(head)
library(ISLR)
?pivot_wider


ggplot()+
  geom_point(trees,aes(x=Girth,y=Height,size=Volume))

trees%>%
  ggplot()+
  geom_point(x=Girth,y=Height,size=Volume)
  
