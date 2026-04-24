# Purpose: Ping-level Exploration for Individual Fish
# Author: Ariel Xing
# Date: 22 Jan 2026
# Contact: ariel.xing@mail.utoronto.ca
# Pre-requisites: processed_AnalysisData_no200
# Any other information needed? None

library(shiny)
library(dplyr)
library(readr)
library(ggplot2)
library(ggiraph)
library(gridExtra)

# ===================== LOAD DATA =====================
load("../../Data/Processed_Old_Data.Rdata")
# ===================== DATA PREPARATION =====================

trackdat <- processed_data_no200 %>% 
  # Keep only Lake Trout (LT) and Smallmouth Bass (SMB)
  filter(grepl("^(LT|SMB)", fishNum)) %>% 
  # Define beam quadrants based on major/minor axis angles
  mutate(
    Quadrat = case_when(
      Angle_major_axis >= 0 & Angle_minor_axis >= 0 ~ "NE",
      Angle_major_axis >= 0 & Angle_minor_axis <  0 ~ "NW",
      Angle_major_axis <  0 & Angle_minor_axis >= 0 ~ "SE",
      Angle_major_axis <  0 & Angle_minor_axis <  0 ~ "SW"
    )
  )

# ===================== UI =====================

ui <- fluidPage(
  titlePanel("Fish Track Ping Exploration"),
  
  sidebarLayout(
    sidebarPanel(
      selectInput(
        inputId  = "fishID",
        label    = strong("Select Fish ID"),
        choices  = sort(unique(trackdat$fishNum)), 
        selected = "LT001"
      ),
      tabPanel("Quadrat Summary", tableOutput("quadTable"))
    ),
    
    mainPanel(
      tabsetPanel(
        
        # ---------- Axis Distances ----------
        tabPanel(
          "Axis Distances",
          girafeOutput("distPlot"),
          tags$p(
            "Each point represents a single acoustic ping from the selected fish. ",
            "The horizontal and vertical axes show the distance of the fish from the ",
            "center of the acoustic beam along the minor (horizontal) and major ",
            "(vertical) axes, respectively, measured in meters (m). ",
            "Point color indicates the mean target strength (TS_mean, in dB) across ",
            "all frequencies for that ping.",
            style = "margin-top: 10px; font-size: 13px; color: #555;"
          )
        ),
        
        # ---------- TS Distribution ----------
        tabPanel(
          "TS Distribution",
          plotOutput("tsHisto"),
          tags$p(
            "Density distributions of mean target strength (TS_mean) for the selected ",
            "fish, shown separately by beam quadrant (NW, NE, SW, SE). ",
            "The gray filled density represents the distribution of TS_mean values for ",
            "the selected individual fish, while the red curve shows the corresponding ",
            "species-level reference distribution aggregated across all individuals ",
            "of the same species.",
            style = "margin-top: 10px; font-size: 13px; color: #555;"
          )
        ),
        
        # ---------- Aspect Angle ----------
        tabPanel(
          "Aspect Angle",
          plotOutput("orientDist"),
          tags$p(
            "Density distributions of fish aspect angle for the selected fish, ",
            "stratified by beam quadrant. The gray filled density corresponds to the ",
            "aspect angle distribution of the selected individual fish, and the red ",
            "curve represents the species-level reference distribution. ",
            "Aspect angle describes the orientation of the fish relative to the ",
            "acoustic beam (positive values indicate head-up orientation, ",
            "negative values indicate head-down orientation).",
            style = "margin-top: 10px; font-size: 13px; color: #555;"
          )
        )
        
      )
    )
  )
)


# ===================== SERVER =====================

server <- function(input, output) {
  
  # ---------- Axis Distances ----------
  output$distPlot <- renderGirafe({
    plotdat <- filter(trackdat, fishNum == input$fishID)
    
    p <- ggplot(plotdat) + 
      geom_point_interactive(
        aes(
          x = Distance_minor_axis,
          y = Distance_major_axis,
          color = TS_mean,
          tooltip = paste("Mean TS:", round(TS_mean, 1), " dB")
        )
      ) +
      geom_vline(xintercept = 0) +
      geom_hline(yintercept = 0) +
      scale_color_viridis_c() +
      labs(
        x = "Distance from beam center (m)",
        y = "Distance from beam center (m)",
        color = "Mean TS (dB)"
      ) +
      theme_bw()
    
    girafe(ggobj = p)
  })
  
  # ---------- TS Distribution (with species-level reference) ----------
  output$tsHisto <- renderPlot({
    
    # Data for the selected individual fish
    fish_dat <- filter(trackdat, fishNum == input$fishID)
    
    # Extract species prefix (e.g., LT001 -> LT, SMB012 -> SMB)
    species_prefix <- sub("[0-9].*$", "", input$fishID)
    
    # Data for all fish of the same species
    species_dat <- trackdat %>%
      filter(grepl(paste0("^", species_prefix), fishNum))
    
    make_ts_plot <- function(q) {
      fish_q    <- filter(fish_dat,    Quadrat == q)
      species_q <- filter(species_dat, Quadrat == q)
      
      ggplot() +
        # Selected fish: gray filled density
        geom_density(
          data  = fish_q,
          aes(x = TS_mean, fill = "Selected fish"),
          alpha = 0.6,
          color = NA
        ) +
        # Species-level reference: red line
        geom_density(
          data  = species_q,
          aes(x = TS_mean, color = "Species overall"),
          size  = 0.9
        ) +
        scale_fill_manual(
          values = c("Selected fish" = "grey80"),
          name   = ""
        ) +
        scale_color_manual(
          values = c("Species overall" = "red"),
          name   = ""
        ) +
        ggtitle(q) +
        labs(
          x = "Mean target strength (dB)",
          y = "Density"
        ) +
        theme_bw() +
        theme(
          legend.position = "bottom"
        )
    }
    
    grid.arrange(
      make_ts_plot("NW"),
      make_ts_plot("NE"),
      make_ts_plot("SW"),
      make_ts_plot("SE")
    )
  })
  
  # ---------- Aspect Angle (with species-level reference) ----------
  output$orientDist <- renderPlot({
    
    # Data for the selected fish
    fish_dat <- filter(trackdat, fishNum == input$fishID)
    
    # Species prefix (LT / SMB)
    species_prefix <- sub("[0-9].*$", "", input$fishID)
    
    # All fish of the same species
    species_dat <- trackdat %>%
      filter(grepl(paste0("^", species_prefix), fishNum))
    
    make_orient_plot <- function(q) {
      fish_q    <- filter(fish_dat,    Quadrat == q)
      species_q <- filter(species_dat, Quadrat == q)
      
      ggplot() +
        # Selected fish: gray filled density
        geom_density(
          data  = fish_q,
          aes(x = aspectAngle, fill = "Selected fish"),
          alpha = 0.6,
          color = NA
        ) +
        # Species-level reference: red line
        geom_density(
          data  = species_q,
          aes(x = aspectAngle, color = "Species overall"),
          size  = 0.9
        ) +
        scale_fill_manual(
          values = c("Selected fish" = "grey80"),
          name   = ""
        ) +
        scale_color_manual(
          values = c("Species overall" = "red"),
          name   = ""
        ) +
        ggtitle(q) +
        labs(
          x = "Aspect angle (degrees)",
          y = "Density"
        ) +
        theme_bw() +
        theme(
          legend.position = "bottom"
        )
    }
    
    grid.arrange(
      make_orient_plot("NW"),
      make_orient_plot("NE"),
      make_orient_plot("SW"),
      make_orient_plot("SE")
    )
  })
  
  # ---------- Quadrat Summary Table ----------
  output$quadTable <- renderTable({
    plotdat <- filter(trackdat, fishNum == input$fishID)
    
    plotdat %>%
      group_by(Quadrat) %>%
      summarize(Ntargets = n(), .groups = "drop")
  })
}

# ===================== RUN APP =====================

shinyApp(ui = ui, server = server)
