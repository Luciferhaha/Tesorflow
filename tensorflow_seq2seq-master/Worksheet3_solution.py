#!/usr/bin/env python3

import math
import timeit

# define a class called "SkipList"
class SkipList:
   start = None
   length = 0

   # Constructor:
   # In Python, the first argument to a constructor is a reference to the object itself
   # ... but this is done automatically (you don't pass this in when calling the constructor)
   # Here, the thing to pass in is a list of elements to store in the skip list
   def __init__( self, elements ):
      size = len(elements)

      sorted_elements = sorted( elements, reverse=True )

      # this is how often a skip pointer should be created
      # ... it is a simple calculation that will not result in 
      #     exactly even skip lengths if the sqrt is not an integer.
      # ... in practice the first skip in the list might be shorter.
      skip_frequency = round( math.sqrt( size ) )
      
      for elem in sorted_elements:
         self.start = SkipNode( elem, self.start )
         if self.length == 0:
            skip_to = self.start
         elif self.length % skip_frequency == 0:
            self.start.skip = skip_to
            skip_to = self.start
         self.length += 1

      self.start.skip = skip_to

# define the nodes that a SkipList will store
class SkipNode:
   next = None    # reference to the next node in the list
   skip = None    # skip pointer: None if it doesn't have a skip pointer, otherwise it's a reference to another node
   element = None # element stored in this node

   # constructor, pass in:
   #   elem:      the element to store in the node
   #   next_node: reference to the next node in the list
   def __init__( self, elem, next_node ):
      self.element = elem
      self.next = next_node


def intersect( list1, list2 ):
   answer = [] # a list to store the answer (for now: we will return a SkipList at the end)
   p1 = list1.start # start the first reference (p1) at the beginning list 1
   p2 = list2.start # start the second reference (p2) at the beginning list 2

   # haven't reached the end of either list
   while p1 is not None and p2 is not None:
      # if p1 and p2 point to the equal elements, add it to the answer and move both pointers
      if p1.element == p2.element:
         answer.append( p1.element )
         p1 = p1.next
         p2 = p2.next
      # p1's element is smaller, so move p1
      elif p1.element < p2.element:
         p1 = p1.next
      # p2's element is smaller, so move p2
      else:
         p2 = p2.next
   return SkipList(answer)

def intersect_skip( list1, list2 ):
   answer = []
   p1 = list1.start
   p2 = list2.start

   while p1 is not None and p2 is not None:
      if p1.element == p2.element:
         answer.append( p1.element )
         p1 = p1.next
         p2 = p2.next
      elif p1.element < p2.element:
         if p1.skip is not None and p1.skip.element < p2.element:
            p1 = p1.skip
         else:
            p1 = p1.next
      else:
         if p2.skip is not None and p2.skip.emement < p1.element:
            p2 = p2.skip
         else:
            p2 = p2.next
   return SkipList(answer)

# good idea to define a function to print the contents of a skip list
def print_skip_list( skiplist ):
   n = skiplist.start
   while n is not None:
      print( n.element, end=' ' )
      n = n.next
   print()

# This is the code that will run when you execute this file with Python.   
if __name__=='__main__':
   list1 = SkipList( range( 0, 100000 ) )
   list2 = SkipList( [ 2, 3, 46, 70, 7222, 999999 ] )

   # check that the output is ok
   print( 'Output for intersect(...)' )
   print_skip_list( intersect( list1, list2 ) )

   print( 'Output for intersect_skip(...)' )
   print_skip_list( intersect_skip( list1, list2 ) )

   # measure the time to run the intersect(...) function
   time_taken = timeit.timeit( 'intersect( list1, list2 )', number = 10, globals = globals() )
   print( 'Execution took {:.4f} seconds'.format( time_taken ) )

   print('Execution took {:.4f} seconds'.format( timeit.timeit( 'intersect_skip( list1, list2 )', number = 10, globals=globals() ) ) )